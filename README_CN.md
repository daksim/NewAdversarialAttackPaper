# Latest Adversarial Attack Papers
**update at 2024-10-08 09:47:04**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Suspiciousness of Adversarial Texts to Human**

对抗性短信对人类的怀疑 cs.LG

Under review

**SubmitDate**: 2024-10-06    [abs](http://arxiv.org/abs/2410.04377v1) [paper-pdf](http://arxiv.org/pdf/2410.04377v1)

**Authors**: Shakila Mahjabin Tonni, Pedro Faustini, Mark Dras

**Abstract**: Adversarial examples pose a significant challenge to deep neural networks (DNNs) across both image and text domains, with the intent to degrade model performance through meticulously altered inputs. Adversarial texts, however, are distinct from adversarial images due to their requirement for semantic similarity and the discrete nature of the textual contents. This study delves into the concept of human suspiciousness, a quality distinct from the traditional focus on imperceptibility found in image-based adversarial examples. Unlike images, where adversarial changes are meant to be indistinguishable to the human eye, textual adversarial content must often remain undetected or non-suspicious to human readers, even when the text's purpose is to deceive NLP systems or bypass filters.   In this research, we expand the study of human suspiciousness by analyzing how individuals perceive adversarial texts. We gather and publish a novel dataset of Likert-scale human evaluations on the suspiciousness of adversarial sentences, crafted by four widely used adversarial attack methods and assess their correlation with the human ability to detect machine-generated alterations. Additionally, we develop a regression-based model to quantify suspiciousness and establish a baseline for future research in reducing the suspiciousness in adversarial text generation. We also demonstrate how the regressor-generated suspicious scores can be incorporated into adversarial generation methods to produce texts that are less likely to be perceived as computer-generated. We make our human suspiciousness annotated data and our code available.

摘要: 对抗性的例子给图像域和文本域的深度神经网络(DNN)带来了巨大的挑战，其意图是通过精心改变输入来降低模型的性能。然而，对抗性文本不同于对抗性图像，因为它们对语义相似性的要求以及文本内容的离散性。这项研究深入探讨了人类猜疑的概念，这一性质有别于传统上对基于图像的对抗性例子中发现的不可察觉的关注。与图像不同的是，对抗性的变化意味着人眼无法区分，而文本的对抗性内容通常必须保持不被发现或对人类读者不可疑，即使文本的目的是欺骗NLP系统或绕过过滤器。在这项研究中，我们通过分析个体如何感知对抗性文本来扩展对人类猜疑的研究。我们收集并发布了一个新的数据集，该数据集由四种广泛使用的对抗性攻击方法创建，并评估了它们与人类检测机器生成的变化的能力之间的相关性。此外，我们开发了一个基于回归的模型来量化可疑性，并为未来在减少对抗性文本生成中的可疑性方面的研究建立了一个基线。我们还演示了如何将回归生成的可疑分数合并到敌意生成方法中，以生成不太可能被视为计算机生成的文本。我们让人类的怀疑、注解的数据和我们的代码可用。



## **2. Adversarial Suffixes May Be Features Too!**

敌对后缀也可能是功能！ cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.00451v2) [paper-pdf](http://arxiv.org/pdf/2410.00451v2)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Despite significant ongoing efforts in safety alignment, large language models (LLMs) such as GPT-4 and LLaMA 3 remain vulnerable to jailbreak attacks that can induce harmful behaviors, including those triggered by adversarial suffixes. Building on prior research, we hypothesize that these adversarial suffixes are not mere bugs but may represent features that can dominate the LLM's behavior. To evaluate this hypothesis, we conduct several experiments. First, we demonstrate that benign features can be effectively made to function as adversarial suffixes, i.e., we develop a feature extraction method to extract sample-agnostic features from benign dataset in the form of suffixes and show that these suffixes may effectively compromise safety alignment. Second, we show that adversarial suffixes generated from jailbreak attacks may contain meaningful features, i.e., appending the same suffix to different prompts results in responses exhibiting specific characteristics. Third, we show that such benign-yet-safety-compromising features can be easily introduced through fine-tuning using only benign datasets, i.e., even in the absence of harmful content. This highlights the critical risk posed by dominating benign features in the training data and calls for further research to reinforce LLM safety alignment. Our code and data is available at \url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.

摘要: 尽管在安全匹配方面正在进行重大的努力，但GPT-4和Llama 3等大型语言模型(LLM)仍然容易受到越狱攻击，这些攻击可能会导致有害行为，包括由对抗性后缀触发的行为。在先前研究的基础上，我们假设这些对抗性后缀不仅仅是错误，而且可能代表可以主导LLM行为的特征。为了评估这一假设，我们进行了几个实验。首先，我们证明了良性特征可以有效地用作对抗性后缀，即，我们开发了一种特征提取方法来从良性数据集中提取与样本无关的后缀形式的特征，并表明这些后缀可以有效地危害安全对齐。其次，我们证明了越狱攻击产生的对抗性后缀可能包含有意义的特征，即在不同的提示后添加相同的后缀会导致响应表现出特定的特征。第三，我们表明，这种良性但危及安全的特征可以通过仅使用良性数据集进行微调来轻松引入，即即使在没有有害内容的情况下也可以。这突出了在训练数据中占主导地位的良性特征所构成的关键风险，并呼吁进一步研究以加强LLM的安全一致性。我们的代码和数据可在\url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.上获得



## **3. Exploring Strengths and Weaknesses of Super-Resolution Attack in Deepfake Detection**

探索Deepfake检测中超分辨率攻击的优点和缺点 cs.CV

Trust What You learN (TWYN) Workshop at European Conference on  Computer Vision ECCV 2024

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04205v1) [paper-pdf](http://arxiv.org/pdf/2410.04205v1)

**Authors**: Davide Alessandro Coccomini, Roberto Caldelli, Fabrizio Falchi, Claudio Gennaro, Giuseppe Amato

**Abstract**: Image manipulation is rapidly evolving, allowing the creation of credible content that can be used to bend reality. Although the results of deepfake detectors are promising, deepfakes can be made even more complicated to detect through adversarial attacks. They aim to further manipulate the image to camouflage deepfakes' artifacts or to insert signals making the image appear pristine. In this paper, we further explore the potential of super-resolution attacks based on different super-resolution techniques and with different scales that can impact the performance of deepfake detectors with more or less intensity. We also evaluated the impact of the attack on more diverse datasets discovering that the super-resolution process is effective in hiding the artifacts introduced by deepfake generation models but fails in hiding the traces contained in fully synthetic images. Finally, we propose some changes to the detectors' training process to improve their robustness to this kind of attack.

摘要: 图像处理正在迅速发展，允许创建可用于扭曲现实的可信内容。尽管Deepfake检测器的结果很有希望，但通过对抗性攻击，Deepfake的检测可能会变得更加复杂。他们的目标是进一步操纵图像以伪装Deepfakes的伪影或插入信号使图像看起来原始。在本文中，我们进一步探讨了基于不同超分辨率技术和不同规模的超分辨率攻击的潜力，这些攻击可能会或多或少地影响Deepfake检测器的性能。我们还评估了攻击对更多样化数据集的影响，发现超分辨率过程可以有效隐藏Deepfake生成模型引入的伪影，但无法隐藏完全合成图像中包含的痕迹。最后，我们对检测器的训练过程提出了一些更改，以提高其对此类攻击的鲁棒性。



## **4. Automated Progressive Red Teaming**

自动化渐进式红色团队 cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2407.03876v2) [paper-pdf](http://arxiv.org/pdf/2407.03876v2)

**Authors**: Bojian Jiang, Yi Jing, Tianhao Shen, Tong Wu, Qing Yang, Deyi Xiong

**Abstract**: Ensuring the safety of large language models (LLMs) is paramount, yet identifying potential vulnerabilities is challenging. While manual red teaming is effective, it is time-consuming, costly and lacks scalability. Automated red teaming (ART) offers a more cost-effective alternative, automatically generating adversarial prompts to expose LLM vulnerabilities. However, in current ART efforts, a robust framework is absent, which explicitly frames red teaming as an effectively learnable task. To address this gap, we propose Automated Progressive Red Teaming (APRT) as an effectively learnable framework. APRT leverages three core modules: an Intention Expanding LLM that generates diverse initial attack samples, an Intention Hiding LLM that crafts deceptive prompts, and an Evil Maker to manage prompt diversity and filter ineffective samples. The three modules collectively and progressively explore and exploit LLM vulnerabilities through multi-round interactions. In addition to the framework, we further propose a novel indicator, Attack Effectiveness Rate (AER) to mitigate the limitations of existing evaluation metrics. By measuring the likelihood of eliciting unsafe but seemingly helpful responses, AER aligns closely with human evaluations. Extensive experiments with both automatic and human evaluations, demonstrate the effectiveness of ARPT across both open- and closed-source LLMs. Specifically, APRT effectively elicits 54% unsafe yet useful responses from Meta's Llama-3-8B-Instruct, 50% from GPT-4o (API access), and 39% from Claude-3.5 (API access), showcasing its robust attack capability and transferability across LLMs (especially from open-source LLMs to closed-source LLMs).

摘要: 确保大型语言模型(LLM)的安全是最重要的，但识别潜在的漏洞是具有挑战性的。虽然手动红色团队是有效的，但它耗时、成本高，而且缺乏可扩展性。自动红色团队(ART)提供了一种更具成本效益的替代方案，可自动生成敌意提示以暴露LLM漏洞。然而，在目前的艺术努力中，缺乏一个强大的框架，它明确地将红色团队作为一项有效的可学习任务。为了弥补这一差距，我们提出了自动渐进红色团队(APRT)作为一种有效的可学习框架。APRT利用三个核心模块：用于生成不同初始攻击样本的意图扩展LLM，用于制作欺骗性提示的意图隐藏LLM，以及用于管理提示多样性和过滤无效样本的邪恶制造者。这三个模块通过多轮交互共同逐步探索和利用LLM漏洞。除了该框架外，我们进一步提出了一个新的指标--攻击效率(AER)，以缓解现有评估指标的局限性。通过衡量引发不安全但似乎有帮助的反应的可能性，AER与人类的评估密切一致。自动和人工评估的广泛实验证明了ARPT在开放源码和封闭源码LLM中的有效性。具体地说，APRT有效地从Meta的Llama-3-8B-Indict、GPT-40(API访问)和Claude-3.5(API访问)中引发了54%的不安全但有用的响应，展示了其强大的攻击能力和跨LLM(特别是从开源LLM到闭源LLM)的可转移性。



## **5. Panda or not Panda? Understanding Adversarial Attacks with Interactive Visualization**

熊猫还是不是熊猫？通过交互式可视化了解对抗性攻击 cs.HC

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2311.13656v2) [paper-pdf](http://arxiv.org/pdf/2311.13656v2)

**Authors**: Yuzhe You, Jarvis Tse, Jian Zhao

**Abstract**: Adversarial machine learning (AML) studies attacks that can fool machine learning algorithms into generating incorrect outcomes as well as the defenses against worst-case attacks to strengthen model robustness. Specifically for image classification, it is challenging to understand adversarial attacks due to their use of subtle perturbations that are not human-interpretable, as well as the variability of attack impacts influenced by diverse methodologies, instance differences, and model architectures. Through a design study with AML learners and teachers, we introduce AdvEx, a multi-level interactive visualization system that comprehensively presents the properties and impacts of evasion attacks on different image classifiers for novice AML learners. We quantitatively and qualitatively assessed AdvEx in a two-part evaluation including user studies and expert interviews. Our results show that AdvEx is not only highly effective as a visualization tool for understanding AML mechanisms, but also provides an engaging and enjoyable learning experience, thus demonstrating its overall benefits for AML learners.

摘要: 对抗性机器学习(AML)研究可以欺骗机器学习算法产生错误结果的攻击，以及对最坏情况下的攻击的防御，以增强模型的健壮性。具体地说，对于图像分类，理解对抗性攻击是具有挑战性的，因为它们使用了人类无法解释的微妙扰动，以及受不同方法、实例差异和模型架构影响的攻击影响的可变性。通过与AML学习者和教师的设计研究，我们介绍了Advex，一个多层次的交互式可视化系统，它为AML初学者全面展示了不同图像分类器上的逃避攻击的特性和影响。我们在包括用户研究和专家访谈的两部分评估中对Advex进行了定量和定性的评估。我们的结果表明，Advex不仅作为一种高效的可视化工具来理解AML的机制，而且还提供了一种引人入胜和愉快的学习体验，从而展示了它对AML学习者的整体好处。



## **6. Adversarial Attacks and Robust Defenses in Speaker Embedding based Zero-Shot Text-to-Speech System**

基于说话人嵌入的零镜头文本到语音系统中的对抗攻击和鲁棒防御 eess.AS

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04017v1) [paper-pdf](http://arxiv.org/pdf/2410.04017v1)

**Authors**: Ze Li, Yao Shi, Yunfei Xu, Ming Li

**Abstract**: Speaker embedding based zero-shot Text-to-Speech (TTS) systems enable high-quality speech synthesis for unseen speakers using minimal data. However, these systems are vulnerable to adversarial attacks, where an attacker introduces imperceptible perturbations to the original speaker's audio waveform, leading to synthesized speech sounds like another person. This vulnerability poses significant security risks, including speaker identity spoofing and unauthorized voice manipulation. This paper investigates two primary defense strategies to address these threats: adversarial training and adversarial purification. Adversarial training enhances the model's robustness by integrating adversarial examples during the training process, thereby improving resistance to such attacks. Adversarial purification, on the other hand, employs diffusion probabilistic models to revert adversarially perturbed audio to its clean form. Experimental results demonstrate that these defense mechanisms can significantly reduce the impact of adversarial perturbations, enhancing the security and reliability of speaker embedding based zero-shot TTS systems in adversarial environments.

摘要: 基于说话人嵌入的零镜头文本到语音(TTS)系统能够使用最少的数据为看不见的说话人进行高质量的语音合成。然而，这些系统容易受到敌意攻击，攻击者在原始说话人的音频波形中引入难以察觉的扰动，导致合成语音听起来像另一个人。该漏洞会带来严重的安全风险，包括演讲者身份欺骗和未经授权的语音操作。本文研究了两种主要的防御策略来应对这些威胁：对抗性训练和对抗性净化。对抗性训练通过在训练过程中整合对抗性实例来增强模型的稳健性，从而提高对此类攻击的抵抗力。另一方面，对抗性净化使用扩散概率模型将对抗性干扰的音频还原为其干净的形式。实验结果表明，这些防御机制能够显著降低对抗性扰动的影响，增强了对抗性环境下基于说话人嵌入的零射击TTS系统的安全性和可靠性。



## **7. Impact of Regularization on Calibration and Robustness: from the Representation Space Perspective**

正规化对校准和鲁棒性的影响：从表示空间的角度来看 cs.CV

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.03999v1) [paper-pdf](http://arxiv.org/pdf/2410.03999v1)

**Authors**: Jonghyun Park, Juyeop Kim, Jong-Seok Lee

**Abstract**: Recent studies have shown that regularization techniques using soft labels, e.g., label smoothing, Mixup, and CutMix, not only enhance image classification accuracy but also improve model calibration and robustness against adversarial attacks. However, the underlying mechanisms of such improvements remain underexplored. In this paper, we offer a novel explanation from the perspective of the representation space (i.e., the space of the features obtained at the penultimate layer). Our investigation first reveals that the decision regions in the representation space form cone-like shapes around the origin after training regardless of the presence of regularization. However, applying regularization causes changes in the distribution of features (or representation vectors). The magnitudes of the representation vectors are reduced and subsequently the cosine similarities between the representation vectors and the class centers (minimal loss points for each class) become higher, which acts as a central mechanism inducing improved calibration and robustness. Our findings provide new insights into the characteristics of the high-dimensional representation space in relation to training and regularization using soft labels.

摘要: 最近的研究表明，使用软标签的正则化技术，如标签平滑、混合和CutMix，不仅提高了图像分类的精度，而且改善了模型校正和对对手攻击的鲁棒性。然而，这种改进的潜在机制仍然没有得到充分的探索。在本文中，我们从表示空间(即在倒数第二层获得的特征的空间)的角度提出了一种新的解释。我们的研究首先发现，无论是否存在正则化，表示空间中的决策区域在训练后都在原点周围形成锥形形状。然而，应用正则化会导致特征(或表示向量)的分布发生变化。表示向量的幅度降低，随后表示向量和类中心(每类的最小损失点)之间的余弦相似度变得更高，这是导致改进的校准和稳健性的中心机制。我们的发现为高维表示空间与使用软标签的训练和正则化相关的特征提供了新的见解。



## **8. Adversarial Attacks on Data Attribution**

对数据归因的对抗性攻击 cs.LG

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2409.05657v2) [paper-pdf](http://arxiv.org/pdf/2409.05657v2)

**Authors**: Xinhe Wang, Pingbang Hu, Junwei Deng, Jiaqi W. Ma

**Abstract**: Data attribution aims to quantify the contribution of individual training data points to the outputs of an AI model, which has been used to measure the value of training data and compensate data providers. Given the impact on financial decisions and compensation mechanisms, a critical question arises concerning the adversarial robustness of data attribution methods. However, there has been little to no systematic research addressing this issue. In this work, we aim to bridge this gap by detailing a threat model with clear assumptions about the adversary's goal and capabilities and proposing principled adversarial attack methods on data attribution. We present two methods, Shadow Attack and Outlier Attack, which generate manipulated datasets to inflate the compensation adversarially. The Shadow Attack leverages knowledge about the data distribution in the AI applications, and derives adversarial perturbations through "shadow training", a technique commonly used in membership inference attacks. In contrast, the Outlier Attack does not assume any knowledge about the data distribution and relies solely on black-box queries to the target model's predictions. It exploits an inductive bias present in many data attribution methods - outlier data points are more likely to be influential - and employs adversarial examples to generate manipulated datasets. Empirically, in image classification and text generation tasks, the Shadow Attack can inflate the data-attribution-based compensation by at least 200%, while the Outlier Attack achieves compensation inflation ranging from 185% to as much as 643%.

摘要: 数据属性旨在量化单个训练数据点对人工智能模型输出的贡献，该模型已被用于衡量训练数据的价值并补偿数据提供者。考虑到对财务决策和补偿机制的影响，数据归因方法的对抗性稳健性出现了一个关键问题。然而，很少或根本没有针对这一问题的系统研究。在这项工作中，我们旨在通过详细描述威胁模型来弥合这一差距，该模型对对手的目标和能力做出了明确的假设，并提出了关于数据属性的原则性对抗性攻击方法。我们提出了两种方法，影子攻击和离群点攻击，这两种方法生成被篡改的数据集来相反地膨胀补偿。影子攻击利用人工智能应用程序中数据分布的知识，通过成员关系推理攻击中常用的一种技术“影子训练”来获得对抗性扰动。相比之下，离群点攻击不假设任何关于数据分布的知识，并且仅依赖于对目标模型的预测的黑盒查询。它利用了许多数据属性方法中存在的归纳偏差--离群值数据点更有可能具有影响力--并使用对抗性例子来生成被操纵的数据集。实验表明，在图像分类和文本生成任务中，阴影攻击可以将基于数据属性的补偿膨胀至少200%，而离群点攻击可以实现185%到高达643%的补偿膨胀。



## **9. Detecting Machine-Generated Long-Form Content with Latent-Space Variables**

检测具有潜在空间变量的机器生成的长形式内容 cs.CL

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03856v1) [paper-pdf](http://arxiv.org/pdf/2410.03856v1)

**Authors**: Yufei Tian, Zeyu Pan, Nanyun Peng

**Abstract**: The increasing capability of large language models (LLMs) to generate fluent long-form texts is presenting new challenges in distinguishing machine-generated outputs from human-written ones, which is crucial for ensuring authenticity and trustworthiness of expressions. Existing zero-shot detectors primarily focus on token-level distributions, which are vulnerable to real-world domain shifts, including different prompting and decoding strategies, and adversarial attacks. We propose a more robust method that incorporates abstract elements, such as event transitions, as key deciding factors to detect machine versus human texts by training a latent-space model on sequences of events or topics derived from human-written texts. In three different domains, machine-generated texts, which are originally inseparable from human texts on the token level, can be better distinguished with our latent-space model, leading to a 31% improvement over strong baselines such as DetectGPT. Our analysis further reveals that, unlike humans, modern LLMs like GPT-4 generate event triggers and their transitions differently, an inherent disparity that helps our method to robustly detect machine-generated texts.

摘要: 大型语言模型(LLM)生成流畅的长文本的能力日益增强，这对区分机器生成的输出和人类书写的输出提出了新的挑战，这对确保表达的真实性和可信度至关重要。现有的零射击检测器主要集中在令牌级分发上，这些分发容易受到现实世界域转换的影响，包括不同的提示和解码策略，以及敌意攻击。我们提出了一种更健壮的方法，通过对来自人类书写的文本的事件或主题序列训练潜在空间模型，将事件转移等抽象元素作为关键决定因素来检测机器文本与人类文本。在三个不同的领域中，机器生成的文本在标记级别上与人类文本密不可分，使用我们的潜在空间模型可以更好地区分它们，导致比DetectGPT等强基线提高31%。我们的分析进一步表明，与人类不同的是，像GPT-4这样的现代LLM以不同的方式生成事件触发器及其转换，这一固有的差异有助于我们的方法稳健地检测机器生成的文本。



## **10. Evaluation of Security of ML-based Watermarking: Copy and Removal Attacks**

基于ML的水印安全性评估：复制和删除攻击 cs.CV

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2409.18211v2) [paper-pdf](http://arxiv.org/pdf/2409.18211v2)

**Authors**: Vitaliy Kinakh, Brian Pulfer, Yury Belousov, Pierre Fernandez, Teddy Furon, Slava Voloshynovskiy

**Abstract**: The vast amounts of digital content captured from the real world or AI-generated media necessitate methods for copyright protection, traceability, or data provenance verification. Digital watermarking serves as a crucial approach to address these challenges. Its evolution spans three generations: handcrafted, autoencoder-based, and foundation model based methods. While the robustness of these systems is well-documented, the security against adversarial attacks remains underexplored. This paper evaluates the security of foundation models' latent space digital watermarking systems that utilize adversarial embedding techniques. A series of experiments investigate the security dimensions under copy and removal attacks, providing empirical insights into these systems' vulnerabilities. All experimental codes and results are available at https://github.com/vkinakh/ssl-watermarking-attacks .

摘要: 从现实世界或人工智能生成的媒体捕获的大量数字内容需要版权保护、可追溯性或数据来源验证的方法。数字水印是解决这些挑战的重要方法。它的演变跨越了三代：手工制作的、基于自动编码器的和基于基础模型的方法。虽然这些系统的稳健性有据可查，但针对对抗性攻击的安全性仍然缺乏充分的研究。本文评估了利用对抗嵌入技术的基础模型潜在空间数字水印系统的安全性。一系列实验研究了复制和删除攻击下的安全维度，为这些系统的漏洞提供了经验见解。所有实验代码和结果均可在https://github.com/vkinakh/ssl-watermarking-attacks上获取。



## **11. RAFT: Realistic Attacks to Fool Text Detectors**

RAFT：愚弄文本检测器的现实攻击 cs.CL

Accepted by EMNLP 2024

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03658v1) [paper-pdf](http://arxiv.org/pdf/2410.03658v1)

**Authors**: James Wang, Ran Li, Junfeng Yang, Chengzhi Mao

**Abstract**: Large language models (LLMs) have exhibited remarkable fluency across various tasks. However, their unethical applications, such as disseminating disinformation, have become a growing concern. Although recent works have proposed a number of LLM detection methods, their robustness and reliability remain unclear. In this paper, we present RAFT: a grammar error-free black-box attack against existing LLM detectors. In contrast to previous attacks for language models, our method exploits the transferability of LLM embeddings at the word-level while preserving the original text quality. We leverage an auxiliary embedding to greedily select candidate words to perturb against the target detector. Experiments reveal that our attack effectively compromises all detectors in the study across various domains by up to 99%, and are transferable across source models. Manual human evaluation studies show our attacks are realistic and indistinguishable from original human-written text. We also show that examples generated by RAFT can be used to train adversarially robust detectors. Our work shows that current LLM detectors are not adversarially robust, underscoring the urgent need for more resilient detection mechanisms.

摘要: 大型语言模型(LLM)在各种任务中表现出了惊人的流畅性。然而，它们不道德的应用，如传播虚假信息，已经成为一个日益令人担忧的问题。虽然最近的工作已经提出了一些LLM检测方法，但它们的稳健性和可靠性仍然不清楚。本文提出了一种针对现有LLM检测器的无语法错误的黑盒攻击方法RAFT。与以往对语言模型的攻击不同，我们的方法在保持原始文本质量的同时，利用了LLM嵌入在单词级别的可转移性。我们利用辅助嵌入来贪婪地选择候选单词来扰动目标检测器。实验表明，我们的攻击有效地危害了研究中跨不同域的所有检测器高达99%，并且可以跨源模型传输。人工人工评估研究表明，我们的攻击是真实的，与原始的人类书面文本没有什么区别。我们还表明，由RAFT生成的例子可以用于训练对抗性稳健的检测器。我们的工作表明，目前的LLM检测器并不具有相反的健壮性，这突显了对更具弹性的检测机制的迫切需要。



## **12. Jailbreaking as a Reward Misspecification Problem**

越狱是奖励错误指定问题 cs.LG

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2406.14393v3) [paper-pdf](http://arxiv.org/pdf/2406.14393v3)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. This misspecification occurs when the reward function fails to accurately capture the intended behavior, leading to misaligned model outputs. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts in a reward-misspecified space. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark against various target aligned LLMs while preserving the human readability of the generated prompts. Furthermore, these attacks on open-source models demonstrate high transferability to closed-source models like GPT-4o and out-of-distribution tasks from HarmBench. Detailed analysis highlights the unique advantages of the proposed reward misspecification objective compared to previous methods, offering new insights for improving LLM safety and robustness.

摘要: 大型语言模型(LLM)的广泛采用引起了人们对它们的安全性和可靠性的担忧，特别是它们对对手攻击的脆弱性。在本文中，我们提出了一种新的观点，将该漏洞归因于对齐过程中的错误指定。当奖励函数未能准确捕获预期行为时，就会出现这种错误说明，从而导致模型输出不对齐。我们引入了一个度量指标ReGap来量化奖励错误指定的程度，并展示了它在检测有害后门提示方面的有效性和健壮性。在这些见解的基础上，我们提出了REMISTY，这是一个用于自动红色团队的系统，它在错误指定奖励的空间中生成对抗性提示。在保持生成提示的人类可读性的同时，针对各种目标对齐的LLM，在AdvBtch基准上实现了最先进的攻击成功率。此外，这些对开源模型的攻击表明，可以很好地转移到GPT-4o等封闭源代码模型和来自HarmBtch的非分发任务。详细的分析强调了与以前的方法相比，所提出的奖励误指定目标的独特优势，为提高LLM的安全性和稳健性提供了新的见解。



## **13. Gradient-based Jailbreak Images for Multimodal Fusion Models**

多模式融合模型的基于对象的越狱图像 cs.CR

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03489v1) [paper-pdf](http://arxiv.org/pdf/2410.03489v1)

**Authors**: Javier Rando, Hannah Korevaar, Erik Brinkman, Ivan Evtimov, Florian Tramèr

**Abstract**: Augmenting language models with image inputs may enable more effective jailbreak attacks through continuous optimization, unlike text inputs that require discrete optimization. However, new multimodal fusion models tokenize all input modalities using non-differentiable functions, which hinders straightforward attacks. In this work, we introduce the notion of a tokenizer shortcut that approximates tokenization with a continuous function and enables continuous optimization. We use tokenizer shortcuts to create the first end-to-end gradient image attacks against multimodal fusion models. We evaluate our attacks on Chameleon models and obtain jailbreak images that elicit harmful information for 72.5% of prompts. Jailbreak images outperform text jailbreaks optimized with the same objective and require 3x lower compute budget to optimize 50x more input tokens. Finally, we find that representation engineering defenses, like Circuit Breakers, trained only on text attacks can effectively transfer to adversarial image inputs.

摘要: 与需要离散优化的文本输入不同，使用图像输入增强语言模型可能会通过持续优化实现更有效的越狱攻击。然而，新的多模式融合模型使用不可微函数来标记化所有输入模式，这阻碍了直接攻击。在这项工作中，我们引入了标记器捷径的概念，它近似于连续函数的标记化，并使连续优化成为可能。我们使用标记器快捷键创建了第一个针对多模式融合模型的端到端梯度图像攻击。我们评估我们对变色龙模型的攻击，并获得72.5%的提示中引发有害信息的越狱图像。越狱图像的性能优于针对相同目标进行优化的文本越狱，并且需要将计算预算降低3倍才能优化50倍以上的输入令牌。最后，我们发现，表示工程防御，如断路器，只接受文本攻击训练，可以有效地转移到对抗性图像输入。



## **14. Mitigating Adversarial Perturbations for Deep Reinforcement Learning via Vector Quantization**

通过载体量化缓解深度强化学习的对抗性扰动 cs.LG

8 pages, IROS 2024 (Code: https://github.com/tunglm2203/vq_robust_rl)

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03376v1) [paper-pdf](http://arxiv.org/pdf/2410.03376v1)

**Authors**: Tung M. Luu, Thanh Nguyen, Tee Joshua Tian Jin, Sungwoon Kim, Chang D. Yoo

**Abstract**: Recent studies reveal that well-performing reinforcement learning (RL) agents in training often lack resilience against adversarial perturbations during deployment. This highlights the importance of building a robust agent before deploying it in the real world. Most prior works focus on developing robust training-based procedures to tackle this problem, including enhancing the robustness of the deep neural network component itself or adversarially training the agent on strong attacks. In this work, we instead study an input transformation-based defense for RL. Specifically, we propose using a variant of vector quantization (VQ) as a transformation for input observations, which is then used to reduce the space of adversarial attacks during testing, resulting in the transformed observations being less affected by attacks. Our method is computationally efficient and seamlessly integrates with adversarial training, further enhancing the robustness of RL agents against adversarial attacks. Through extensive experiments in multiple environments, we demonstrate that using VQ as the input transformation effectively defends against adversarial attacks on the agent's observations.

摘要: 最近的研究表明，在训练中表现良好的强化学习(RL)代理在部署过程中往往缺乏对对手扰动的弹性。这突显了在将其部署到现实世界之前构建一个强大的代理的重要性。以前的大多数工作都集中在开发健壮的基于训练的过程来解决这个问题，包括增强深度神经网络组件本身的健壮性或相反地训练代理进行强攻击。在这项工作中，我们转而研究基于输入变换的RL防御。具体地说，我们提出使用矢量量化(VQ)的一种变体作为输入观测的变换，然后使用该变换来减小测试过程中敌对攻击的空间，从而使得变换后的观测受到攻击的影响较小。我们的方法计算效率高，并与对抗性训练无缝结合，进一步增强了RL代理对对抗性攻击的健壮性。通过在多个环境中的广泛实验，我们证明了使用VQ作为输入变换有效地防御了对主体观察的敌意攻击。



## **15. The Vital Role of Gradient Clipping in Byzantine-Resilient Distributed Learning**

梯度剪辑在拜占庭弹性分布式学习中的重要作用 cs.LG

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2405.14432v3) [paper-pdf](http://arxiv.org/pdf/2405.14432v3)

**Authors**: Youssef Allouah, Rachid Guerraoui, Nirupam Gupta, Ahmed Jellouli, Geovani Rizk, John Stephan

**Abstract**: Byzantine-resilient distributed machine learning seeks to achieve robust learning performance in the presence of misbehaving or adversarial workers. While state-of-the-art (SOTA) robust distributed gradient descent (Robust-DGD) methods were proven theoretically optimal, their empirical success has often relied on pre-aggregation gradient clipping. However, the currently considered static clipping strategy exhibits mixed results: improving robustness against some attacks while being ineffective or detrimental against others. We address this gap by proposing a principled adaptive clipping strategy, termed Adaptive Robust Clipping (ARC). We show that ARC consistently enhances the empirical robustness of SOTA Robust-DGD methods, while preserving the theoretical robustness guarantees. Our analysis shows that ARC provably improves the asymptotic convergence guarantee of Robust-DGD in the case when the model is well-initialized. We validate this theoretical insight through an exhaustive set of experiments on benchmark image classification tasks. We observe that the improvement induced by ARC is more pronounced in highly heterogeneous and adversarial settings.

摘要: 拜占庭-弹性分布式机器学习寻求在存在行为不端或敌对的工作人员的情况下实现稳健的学习性能。虽然最先进的(SOTA)稳健分布梯度下降(Robust-DGD)方法在理论上被证明是最优的，但它们的经验成功通常依赖于预聚聚梯度裁剪。然而，目前考虑的静态裁剪策略呈现出好坏参半的结果：提高了对某些攻击的健壮性，而对另一些攻击无效或有害。我们提出了一种原则性的自适应剪裁策略，称为自适应稳健剪裁(ARC)，以解决这一差距。我们证明了ARC在保持理论稳健性保证的同时，一致地增强了SOTA Robust-DGD方法的经验稳健性。我们的分析表明，在模型良好初始化的情况下，ARC明显改善了Robust-DGD的渐近收敛保证。我们通过一组详尽的基准图像分类任务的实验来验证这一理论见解。我们观察到，在高度异质性和对抗性的环境中，ARC带来的改善更加显著。



## **16. Adversarial Challenges in Network Intrusion Detection Systems: Research Insights and Future Prospects**

网络入侵检测系统中的对抗挑战：研究见解和未来前景 cs.CR

35 pages

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2409.18736v2) [paper-pdf](http://arxiv.org/pdf/2409.18736v2)

**Authors**: Sabrine Ennaji, Fabio De Gaspari, Dorjan Hitaj, Alicia K Bidi, Luigi V. Mancini

**Abstract**: Machine learning has brought significant advances in cybersecurity, particularly in the development of Intrusion Detection Systems (IDS). These improvements are mainly attributed to the ability of machine learning algorithms to identify complex relationships between features and effectively generalize to unseen data. Deep neural networks, in particular, contributed to this progress by enabling the analysis of large amounts of training data, significantly enhancing detection performance. However, machine learning models remain vulnerable to adversarial attacks, where carefully crafted input data can mislead the model into making incorrect predictions. While adversarial threats in unstructured data, such as images and text, have been extensively studied, their impact on structured data like network traffic is less explored. This survey aims to address this gap by providing a comprehensive review of machine learning-based Network Intrusion Detection Systems (NIDS) and thoroughly analyzing their susceptibility to adversarial attacks. We critically examine existing research in NIDS, highlighting key trends, strengths, and limitations, while identifying areas that require further exploration. Additionally, we discuss emerging challenges in the field and offer insights for the development of more robust and resilient NIDS. In summary, this paper enhances the understanding of adversarial attacks and defenses in NIDS and guide future research in improving the robustness of machine learning models in cybersecurity applications.

摘要: 机器学习在网络安全方面带来了重大进展，特别是在入侵检测系统(入侵检测系统)的开发方面。这些改进主要归功于机器学习算法能够识别特征之间的复杂关系，并有效地概括到看不见的数据。尤其是深度神经网络，通过能够分析大量训练数据，大大提高了检测性能，从而促进了这一进展。然而，机器学习模型仍然容易受到敌意攻击，精心设计的输入数据可能会误导模型做出错误的预测。尽管图像和文本等非结构化数据中的敌意威胁已被广泛研究，但它们对网络流量等结构化数据的影响却鲜有人探讨。这项调查旨在通过对基于机器学习的网络入侵检测系统(NID)的全面审查来解决这一差距，并彻底分析它们对对手攻击的敏感性。我们批判性地检查NID中的现有研究，强调主要趋势、优势和局限性，同时确定需要进一步探索的领域。此外，我们还讨论了该领域新出现的挑战，并为开发更强大和更具弹性的网络入侵检测系统提供了见解。综上所述，本文加深了对网络入侵检测系统中对抗性攻击和防御的理解，并指导了未来在提高机器学习模型在网络安全应用中的稳健性方面的研究。



## **17. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

SCA：高效语义一致的无限制对抗攻击 cs.CV

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.02240v2) [paper-pdf](http://arxiv.org/pdf/2410.02240v2)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our code can be found at https://github.com/Pan-Zihao/SCA.

摘要: 不受限制的对抗性攻击通常操纵图像的语义内容(例如，颜色或纹理)以创建既有效又逼真的对抗性示例。最近的工作利用扩散逆过程将图像映射到潜在空间，在潜在空间中通过引入扰动来操纵高级语义。然而，它们往往会在去噪输出中造成严重的语义扭曲，并导致效率低下。在这项研究中，我们提出了一种新的框架，称为语义一致的无限对抗攻击(SCA)，它使用一种反转方法来提取编辑友好的噪声映射，并利用多模式大语言模型(MLLM)在整个过程中提供语义指导。在MLLM提供丰富语义信息的条件下，使用一系列编辑友好的噪声图对每个步骤进行DDPM去噪处理，并利用DPM Solver++加速这一过程，从而实现高效的语义一致性采样。与现有的方法相比，我们的框架能够高效地生成对抗性的例子，这些例子表现出最小的可识别的语义变化。因此，我们首次引入了语义一致的对抗性例子(SCAE)。广泛的实验和可视化已经证明了SCA的高效率，特别是在平均速度上是最先进的攻击的12倍。我们的代码可以在https://github.com/Pan-Zihao/SCA.上找到



## **18. MITS-GAN: Safeguarding Medical Imaging from Tampering with Generative Adversarial Networks**

MITS-GAN：保护医学成像免受生成性对抗网络的篡改 eess.IV

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2401.09624v2) [paper-pdf](http://arxiv.org/pdf/2401.09624v2)

**Authors**: Giovanni Pasqualino, Luca Guarnera, Alessandro Ortis, Sebastiano Battiato

**Abstract**: The progress in generative models, particularly Generative Adversarial Networks (GANs), opened new possibilities for image generation but raised concerns about potential malicious uses, especially in sensitive areas like medical imaging. This study introduces MITS-GAN, a novel approach to prevent tampering in medical images, with a specific focus on CT scans. The approach disrupts the output of the attacker's CT-GAN architecture by introducing finely tuned perturbations that are imperceptible to the human eye. Specifically, the proposed approach involves the introduction of appropriate Gaussian noise to the input as a protective measure against various attacks. Our method aims to enhance tamper resistance, comparing favorably to existing techniques. Experimental results on a CT scan demonstrate MITS-GAN's superior performance, emphasizing its ability to generate tamper-resistant images with negligible artifacts. As image tampering in medical domains poses life-threatening risks, our proactive approach contributes to the responsible and ethical use of generative models. This work provides a foundation for future research in countering cyber threats in medical imaging. Models and codes are publicly available on https://iplab.dmi.unict.it/MITS-GAN-2024/.

摘要: 生成性模型的进展，特别是生成性对抗网络(GANS)，为图像生成打开了新的可能性，但也引发了人们对潜在恶意使用的担忧，特别是在医学成像等敏感领域。这项研究介绍了一种新的防止医学图像篡改的方法MITS-GaN，重点介绍了CT扫描。这种方法通过引入人眼无法察觉的微调微扰，扰乱了攻击者的CT-GaN体系结构的输出。具体地说，所提出的方法包括在输入中引入适当的高斯噪声作为对各种攻击的保护措施。我们的方法旨在增强抗篡改能力，与现有技术相比是有利的。CT扫描的实验结果显示了MITS-GaN的优越性能，强调了其生成具有可忽略伪影的防篡改图像的能力。由于医学领域的图像篡改带来了危及生命的风险，我们积极主动的方法有助于负责任和合乎道德地使用生殖模型。这项工作为未来在医学成像中对抗网络威胁的研究提供了基础。型号和代码在https://iplab.dmi.unict.it/MITS-GAN-2024/.上公开提供



## **19. Prevailing against Adversarial Noncentral Disturbances: Exact Recovery of Linear Systems with the $l_1$-norm Estimator**

对抗非中心扰动：用$l_1$-模估计精确恢复线性系统 math.OC

8 pages, 2 figures

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03218v1) [paper-pdf](http://arxiv.org/pdf/2410.03218v1)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper studies the linear system identification problem in the general case where the disturbance is sub-Gaussian, correlated, and possibly adversarial. First, we consider the case with noncentral (nonzero-mean) disturbances for which the ordinary least-squares (OLS) method fails to correctly identify the system. We prove that the $l_1$-norm estimator accurately identifies the system under the condition that each disturbance has equal probabilities of being positive or negative. This condition restricts the sign of each disturbance but allows its magnitude to be arbitrary. Second, we consider the case where each disturbance is adversarial with the model that the attack times happen occasionally but the distributions of the attack values are completely arbitrary. We show that when the probability of having an attack at a given time is less than 0.5, the $l_1$-norm estimator prevails against any adversarial noncentral disturbances and the exact recovery is achieved within a finite time. These results pave the way to effectively defend against arbitrarily large noncentral attacks in safety-critical systems.

摘要: 本文研究一般情况下的线性系统辨识问题，其中扰动是亚高斯的，相关的，可能是对抗性的。首先，我们考虑了具有非中心(非零均值)扰动的情况，对于这种情况，普通的最小二乘(OLS)方法不能正确地辨识系统。我们证明了在每个扰动具有相等的正负概率的条件下，$L_1$-范数估计量能够准确地辨识系统。这一条件限制了每个扰动的符号，但允许其大小任意。其次，在攻击次数偶尔发生但攻击值的分布完全任意的情况下，我们考虑了每次扰动是对抗性的情况。我们证明了当给定时刻发生攻击的概率小于0.5时，$L_1$-范数估计对任何对抗性非中心扰动都是有效的，并且在有限时间内实现了精确的恢复。这些结果为在安全关键系统中有效防御任意规模的非中心攻击铺平了道路。



## **20. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

多语言机器生成文本检测中的作者混淆 cs.CL

Accepted to EMNLP 2024 Findings

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2401.07867v3) [paper-pdf](http://arxiv.org/pdf/2401.07867v3)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of recent Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause evasion of automated detection in all tested languages, where homoglyph attacks are especially successful. However, some of the AO methods severely damaged the text, making it no longer readable or easily recognizable by humans (e.g., changed language, weird characters).

摘要: 最近的大型语言模型(LLM)的高质量文本生成能力引起了人们对它们的滥用(例如，在大规模生成/传播虚假信息中)的担忧。机器生成文本(MGT)检测对于应对此类威胁非常重要。然而，它容易受到作者身份混淆(AO)方法的影响，例如转译，这可能导致MGTS逃避检测。到目前为止，这只在单一语言环境中进行了评估。因此，最近提出的多语言检测器的敏感性仍然未知。我们通过全面基准测试10种著名的AO方法的性能来填补这一空白，针对11种语言的MGT攻击37种MGT检测方法(即，10$\乘以$37$\乘以$11=4,070个组合)。我们还使用混淆文本来评估数据增强对对手健壮性的影响。结果表明，在所有被测语言中，所有被测试的声学方法都可以逃避自动检测，其中同形文字攻击尤其成功。然而，一些AO方法严重损坏了文本，使其不再可读或不再容易被人类识别(例如，改变语言、奇怪的字符)。



## **21. Investigating Imperceptibility of Adversarial Attacks on Tabular Data: An Empirical Analysis**

调查表格数据对抗性攻击的不可感知性：实证分析 cs.LG

36 pages

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2407.11463v3) [paper-pdf](http://arxiv.org/pdf/2407.11463v3)

**Authors**: Zhipeng He, Chun Ouyang, Laith Alzubaidi, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks are a potential threat to machine learning models by causing incorrect predictions through imperceptible perturbations to the input data. While these attacks have been extensively studied in unstructured data like images, applying them to tabular data, poses new challenges. These challenges arise from the inherent heterogeneity and complex feature interdependencies in tabular data, which differ from the image data. To account for this distinction, it is necessary to establish tailored imperceptibility criteria specific to tabular data. However, there is currently a lack of standardised metrics for assessing the imperceptibility of adversarial attacks on tabular data. To address this gap, we propose a set of key properties and corresponding metrics designed to comprehensively characterise imperceptible adversarial attacks on tabular data. These are: proximity to the original input, sparsity of altered features, deviation from the original data distribution, sensitivity in perturbing features with narrow distribution, immutability of certain features that should remain unchanged, feasibility of specific feature values that should not go beyond valid practical ranges, and feature interdependencies capturing complex relationships between data attributes. We evaluate the imperceptibility of five adversarial attacks, including both bounded attacks and unbounded attacks, on tabular data using the proposed imperceptibility metrics. The results reveal a trade-off between the imperceptibility and effectiveness of these attacks. The study also identifies limitations in current attack algorithms, offering insights that can guide future research in the area. The findings gained from this empirical analysis provide valuable direction for enhancing the design of adversarial attack algorithms, thereby advancing adversarial machine learning on tabular data.

摘要: 对抗性攻击通过对输入数据的不可察觉的扰动而导致错误的预测，从而对机器学习模型构成潜在的威胁。虽然这些攻击已经在图像等非结构化数据中得到了广泛研究，但将它们应用于表格数据带来了新的挑战。这些挑战源于表格数据固有的异构性和复杂的特征相互依赖关系，而表格数据不同于图像数据。为了说明这一区别，有必要建立专门针对表格数据的不可察觉标准。然而，目前缺乏用于评估对抗性攻击对表格数据的不可感知性的标准化指标。为了弥补这一差距，我们提出了一组关键属性和相应的度量，旨在全面表征对表格数据的不可察觉的对抗性攻击。它们是：接近原始输入、改变特征的稀疏性、偏离原始数据分布、对具有窄分布的扰动特征的敏感性、某些应保持不变的特征的不变性、不应超出有效实际范围的特定特征值的可行性、以及捕捉数据属性之间的复杂关系的特征相互依赖关系。我们使用所提出的不可感知性度量评估了五种对抗性攻击，包括有界攻击和无界攻击对表格数据的不可感知性。结果揭示了这些攻击的隐蔽性和有效性之间的权衡。该研究还确定了当前攻击算法的局限性，提供了可以指导该领域未来研究的见解。这一实证分析的结果为改进对抗性攻击算法的设计，从而推进对抗性表格数据机器学习提供了有价值的指导。



## **22. EIA: Environmental Injection Attack on Generalist Web Agents for Privacy Leakage**

EIA：针对多面手网络代理隐私泄露的环境注入攻击 cs.CR

29 pages

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2409.11295v3) [paper-pdf](http://arxiv.org/pdf/2409.11295v3)

**Authors**: Zeyi Liao, Lingbo Mo, Chejian Xu, Mintong Kang, Jiawei Zhang, Chaowei Xiao, Yuan Tian, Bo Li, Huan Sun

**Abstract**: Generalist web agents have demonstrated remarkable potential in autonomously completing a wide range of tasks on real websites, significantly boosting human productivity. However, web tasks, such as booking flights, usually involve users' PII, which may be exposed to potential privacy risks if web agents accidentally interact with compromised websites, a scenario that remains largely unexplored in the literature. In this work, we narrow this gap by conducting the first study on the privacy risks of generalist web agents in adversarial environments. First, we present a realistic threat model for attacks on the website, where we consider two adversarial targets: stealing users' specific PII or the entire user request. Then, we propose a novel attack method, termed Environmental Injection Attack (EIA). EIA injects malicious content designed to adapt well to environments where the agents operate and our work instantiates EIA specifically for privacy scenarios in web environments. We collect 177 action steps that involve diverse PII categories on realistic websites from the Mind2Web, and conduct experiments using one of the most capable generalist web agent frameworks to date. The results demonstrate that EIA achieves up to 70% ASR in stealing specific PII and 16% ASR for full user request. Additionally, by accessing the stealthiness and experimenting with a defensive system prompt, we indicate that EIA is hard to detect and mitigate. Notably, attacks that are not well adapted for a webpage can be detected via human inspection, leading to our discussion about the trade-off between security and autonomy. However, extra attackers' efforts can make EIA seamlessly adapted, rendering such supervision ineffective. Thus, we further discuss the defenses at the pre- and post-deployment stages of the websites without relying on human supervision and call for more advanced defense strategies.

摘要: 多面手网络代理在自主完成真实网站上的各种任务方面表现出了非凡的潜力，显著提高了人类的生产力。然而，预订机票等网络任务通常涉及用户的PII，如果网络代理意外地与受影响的网站交互，可能会面临潜在的隐私风险，这种情况在文献中基本上仍未探讨。在这项工作中，我们通过对对抗环境中通才网络代理的隐私风险进行第一次研究来缩小这一差距。首先，我们给出了一个现实的网站攻击威胁模型，其中我们考虑了两个敌对目标：窃取用户的特定PII或整个用户请求。然后，我们提出了一种新的攻击方法，称为环境注入攻击(EIA)。EIA注入恶意内容，旨在很好地适应代理运行的环境，我们的工作特别针对Web环境中的隐私场景实例化了EIA。我们从Mind2Web收集了177个动作步骤，涉及现实网站上不同的PII类别，并使用迄今最有能力的通才Web代理框架之一进行了实验。结果表明，EIA在窃取特定PII请求时获得了高达70%的ASR，对于完整的用户请求达到了16%的ASR。此外，通过访问隐蔽性和试验防御系统提示，我们表明EIA很难检测和缓解。值得注意的是，没有很好地适应网页的攻击可以通过人工检查来检测，这导致了我们关于安全性和自主性之间的权衡的讨论。然而，额外的攻击者的努力可能会使EIA无缝适应，使这种监督无效。因此，我们进一步讨论了网站部署前和部署后阶段的防御，而不依赖于人的监督，并呼吁更先进的防御策略。



## **23. Influence-based Attributions can be Manipulated**

基于影响力的归因可以被操纵 cs.LG

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2409.05208v4) [paper-pdf](http://arxiv.org/pdf/2409.05208v4)

**Authors**: Chhavi Yadav, Ruihan Wu, Kamalika Chaudhuri

**Abstract**: Influence Functions are a standard tool for attributing predictions to training data in a principled manner and are widely used in applications such as data valuation and fairness. In this work, we present realistic incentives to manipulate influence-based attributions and investigate whether these attributions can be \textit{systematically} tampered by an adversary. We show that this is indeed possible for logistic regression models trained on ResNet feature embeddings and standard tabular fairness datasets and provide efficient attacks with backward-friendly implementations. Our work raises questions on the reliability of influence-based attributions in adversarial circumstances. Code is available at : \url{https://github.com/infinite-pursuits/influence-based-attributions-can-be-manipulated}

摘要: 影响力函数是一种标准工具，用于以有原则的方式将预测归因于训练数据，并广泛用于数据评估和公平性等应用中。在这项工作中，我们提出了操纵基于影响力的属性的现实激励，并调查这些属性是否可以被对手\texttit {系统性地}篡改。我们表明，对于在ResNet特征嵌入和标准表格公平性数据集上训练的逻辑回归模型来说，这确实是可能的，并通过向后友好的实现提供有效的攻击。我们的工作对敌对情况下基于影响力的归因的可靠性提出了质疑。代码可在：\url{https：//github.com/infinite-purpers/influence-based-attributions-can-be-manifolded}



## **24. Kick Bad Guys Out! Conditionally Activated Anomaly Detection in Federated Learning with Zero-Knowledge Proof Verification**

把坏人踢出去！具有零知识证明验证的联邦学习中的一致激活异常检测 cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2310.04055v3) [paper-pdf](http://arxiv.org/pdf/2310.04055v3)

**Authors**: Shanshan Han, Wenxuan Wu, Baturalp Buyukates, Weizhao Jin, Qifan Zhang, Yuhang Yao, Salman Avestimehr, Chaoyang He

**Abstract**: Federated Learning (FL) systems are susceptible to adversarial attacks, where malicious clients submit poisoned models to disrupt the convergence or plant backdoors that cause the global model to misclassify some samples. Current defense methods are often impractical for real-world FL systems, as they either rely on unrealistic prior knowledge or cause accuracy loss even in the absence of attacks. Furthermore, these methods lack a protocol for verifying execution, leaving participants uncertain about the correct execution of the mechanism. To address these challenges, we propose a novel anomaly detection strategy that is designed for real-world FL systems. Our approach activates the defense only when potential attacks are detected, and enables the removal of malicious models without affecting the benign ones. Additionally, we incorporate zero-knowledge proofs to ensure the integrity of the proposed defense mechanism. Experimental results demonstrate the effectiveness of our approach in enhancing FL system security against a comprehensive set of adversarial attacks in various ML tasks.

摘要: 联邦学习(FL)系统容易受到敌意攻击，恶意客户端提交有毒模型来扰乱收敛或植入后门，导致全局模型对某些样本进行错误分类。目前的防御方法对于真实的FL系统来说往往是不切实际的，因为它们要么依赖不切实际的先验知识，要么即使在没有攻击的情况下也会造成准确性损失。此外，这些方法缺乏验证执行的协议，使得参与者不确定该机制的正确执行。为了应对这些挑战，我们提出了一种新的异常检测策略，该策略是为现实世界的FL系统设计的。我们的方法只有在检测到潜在攻击时才会激活防御，并且可以在不影响良性模型的情况下删除恶意模型。此外，我们加入了零知识证明，以确保所提出的防御机制的完整性。实验结果表明，该方法能有效地提高FL系统在各种ML任务中抵抗各种敌意攻击的安全性。



## **25. Safeguard is a Double-edged Sword: Denial-of-service Attack on Large Language Models**

保障是一把双刃剑：对大型语言模型的拒绝服务攻击 cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02916v1) [paper-pdf](http://arxiv.org/pdf/2410.02916v1)

**Authors**: Qingzhao Zhang, Ziyang Xiong, Z. Morley Mao

**Abstract**: Safety is a paramount concern of large language models (LLMs) in their open deployment. To this end, safeguard methods aim to enforce the ethical and responsible use of LLMs through safety alignment or guardrail mechanisms. However, we found that the malicious attackers could exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a new denial-of-service (DoS) attack on LLMs. Specifically, by software or phishing attacks on user client software, attackers insert a short, seemingly innocuous adversarial prompt into to user prompt templates in configuration files; thus, this prompt appears in final user requests without visibility in the user interface and is not trivial to identify. By designing an optimization process that utilizes gradient and attention information, our attack can automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97\% of user requests on Llama Guard 3. The attack presents a new dimension of evaluating LLM safeguards focusing on false positives, fundamentally different from the classic jailbreak.

摘要: 安全是大型语言模型(LLM)在开放部署时最关心的问题。为此，保障措施旨在通过安全调整或护栏机制，强制以合乎道德和负责任的方式使用LLMS。然而，我们发现恶意攻击者可以利用安全措施的误报，即欺骗安全措施模型错误地阻止安全内容，从而导致对LLMS的新的拒绝服务(DoS)攻击。具体地说，通过软件或对用户客户端软件的网络钓鱼攻击，攻击者将一个看似无害的简短对抗性提示插入到配置文件中的用户提示模板中；因此，该提示出现在最终用户请求中，在用户界面中不可见，并且很难识别。通过设计一个利用梯度和注意力信息的优化过程，我们的攻击可以自动生成看似安全的敌意提示，大约只有30个字符，普遍阻止Llama Guard 3上超过97%的用户请求。该攻击提供了一个新的维度来评估LLM安全措施，从根本上不同于传统的越狱。



## **26. Universally Optimal Watermarking Schemes for LLMs: from Theory to Practice**

LLM的普遍最优水印方案：从理论到实践 cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02890v1) [paper-pdf](http://arxiv.org/pdf/2410.02890v1)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Large Language Models (LLMs) boosts human efficiency but also poses misuse risks, with watermarking serving as a reliable method to differentiate AI-generated content from human-created text. In this work, we propose a novel theoretical framework for watermarking LLMs. Particularly, we jointly optimize both the watermarking scheme and detector to maximize detection performance, while controlling the worst-case Type-I error and distortion in the watermarked text. Within our framework, we characterize the universally minimum Type-II error, showing a fundamental trade-off between detection performance and distortion. More importantly, we identify the optimal type of detectors and watermarking schemes. Building upon our theoretical analysis, we introduce a practical, model-agnostic and computationally efficient token-level watermarking algorithm that invokes a surrogate model and the Gumbel-max trick. Empirical results on Llama-13B and Mistral-8$\times$7B demonstrate the effectiveness of our method. Furthermore, we also explore how robustness can be integrated into our theoretical framework, which provides a foundation for designing future watermarking systems with improved resilience to adversarial attacks.

摘要: 大语言模型(LLM)提高了人类的效率，但也带来了滥用风险，水印是区分人工智能生成的内容和人类创建的文本的可靠方法。在这项工作中，我们提出了一种新的水印LLMS的理论框架。特别是，我们联合优化了水印方案和检测器以最大化检测性能，同时控制了最坏情况下的I类错误和水印文本中的失真。在我们的框架内，我们描述了普遍最小的第二类错误，显示了检测性能和失真之间的基本权衡。更重要的是，我们确定了检测器和水印方案的最佳类型。在理论分析的基础上，我们介绍了一种实用的、与模型无关的、计算高效的令牌级水印算法，该算法调用了代理模型和Gumbel-Max技巧。对Llama-13B和Mistral-8$乘以$70B的实验结果证明了该方法的有效性。此外，我们还探索了如何将稳健性融入到我们的理论框架中，这为设计未来具有更好的抗攻击能力的水印系统提供了基础。



## **27. Mitigating Dialogue Hallucination for Large Vision Language Models via Adversarial Instruction Tuning**

通过对抗性指令调优缓解大视野语言模型的对话幻觉 cs.CV

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2403.10492v3) [paper-pdf](http://arxiv.org/pdf/2403.10492v3)

**Authors**: Dongmin Park, Zhaofang Qian, Guangxing Han, Ser-Nam Lim

**Abstract**: Mitigating hallucinations of Large Vision Language Models,(LVLMs) is crucial to enhance their reliability for general-purpose assistants. This paper shows that such hallucinations of LVLMs can be significantly exacerbated by preceding user-system dialogues. To precisely measure this, we first present an evaluation benchmark by extending popular multi-modal benchmark datasets with prepended hallucinatory dialogues powered by our novel Adversarial Question Generator (AQG), which can automatically generate image-related yet adversarial dialogues by adopting adversarial attacks on LVLMs. On our benchmark, the zero-shot performance of state-of-the-art LVLMs drops significantly for both the VQA and Captioning tasks. Next, we further reveal this hallucination is mainly due to the prediction bias toward preceding dialogues rather than visual content. To reduce this bias, we propose Adversarial Instruction Tuning (AIT) that robustly fine-tunes LVLMs against hallucinatory dialogues. Extensive experiments show our proposed approach successfully reduces dialogue hallucination while maintaining performance.

摘要: 减轻大型视觉语言模型(LVLMS)的幻觉对于提高其对通用助理的可靠性至关重要。这篇论文表明，之前的用户-系统对话可以显著加剧LVLMS的这种幻觉。为了准确地衡量这一点，我们首先提出了一个评估基准，通过扩展流行的多模式基准数据集，在我们的新型对抗性问题生成器(AQG)的支持下，使用预先设定的幻觉对话，该生成器可以通过对LVLM进行对抗性攻击来自动生成与图像相关的对抗性对话。在我们的基准测试中，最先进的LVLMS在VQA和字幕任务中的零镜头性能都显著下降。接下来，我们进一步揭示这种幻觉主要是由于预测偏向于之前的对话而不是视觉内容。为了减少这种偏差，我们提出了对抗性指令调整(AIT)，它针对幻觉对话对LVLM进行强有力的微调。大量的实验表明，我们提出的方法在保持性能的同时成功地减少了对话幻觉。



## **28. Erasing Conceptual Knowledge from Language Models**

从语言模型中删除概念知识 cs.CL

Project Page: https://elm.baulab.info

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02760v1) [paper-pdf](http://arxiv.org/pdf/2410.02760v1)

**Authors**: Rohit Gandikota, Sheridan Feucht, Samuel Marks, David Bau

**Abstract**: Concept erasure in language models has traditionally lacked a comprehensive evaluation framework, leading to incomplete assessments of effectiveness of erasure methods. We propose an evaluation paradigm centered on three critical criteria: innocence (complete knowledge removal), seamlessness (maintaining conditional fluent generation), and specificity (preserving unrelated task performance). Our evaluation metrics naturally motivate the development of Erasure of Language Memory (ELM), a new method designed to address all three dimensions. ELM employs targeted low-rank updates to alter output distributions for erased concepts while preserving overall model capabilities including fluency when prompted for an erased concept. We demonstrate ELM's efficacy on biosecurity, cybersecurity, and literary domain erasure tasks. Comparative analysis shows that ELM achieves superior performance across our proposed metrics, including near-random scores on erased topic assessments, generation fluency, maintained accuracy on unrelated benchmarks, and robustness under adversarial attacks. Our code, data, and trained models are available at https://elm.baulab.info

摘要: 传统上，语言模型中的概念删除缺乏一个全面的评估框架，导致对删除方法的有效性的评估不完整。我们提出了一个以三个关键标准为核心的评估范式：清白(完全去除知识)、无缝(保持条件流畅生成)和专一性(保持无关的任务绩效)。我们的评估指标自然推动了语言记忆擦除(ELM)的发展，这是一种旨在解决所有这三个维度的新方法。ELM使用有针对性的低阶更新来改变已擦除概念的输出分布，同时保留总体模型能力，包括在提示输入已擦除概念时的流畅性。我们展示了ELM在生物安全、网络安全和文学领域擦除任务中的有效性。对比分析表明，ELM在我们提出的指标上都取得了优异的性能，包括在擦除主题评估上的近乎随机的分数、生成流畅度、在无关基准上保持的准确性以及在对手攻击下的健壮性。我们的代码、数据和经过培训的模型可在https://elm.baulab.info上获得



## **29. Does Refusal Training in LLMs Generalize to the Past Tense?**

LLM中的拒绝培训是否适用于过去时态？ cs.CL

Update in v3: o1-mini and o1-preview results (on top of GPT-4o and  Claude 3.5 Sonnet added in v2). We provide code and jailbreak artifacts at  https://github.com/tml-epfl/llm-past-tense

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2407.11969v3) [paper-pdf](http://arxiv.org/pdf/2407.11969v3)

**Authors**: Maksym Andriushchenko, Nicolas Flammarion

**Abstract**: Refusal training is widely used to prevent LLMs from generating harmful, undesirable, or illegal outputs. We reveal a curious generalization gap in the current refusal training approaches: simply reformulating a harmful request in the past tense (e.g., "How to make a Molotov cocktail?" to "How did people make a Molotov cocktail?") is often sufficient to jailbreak many state-of-the-art LLMs. We systematically evaluate this method on Llama-3 8B, Claude-3.5 Sonnet, GPT-3.5 Turbo, Gemma-2 9B, Phi-3-Mini, GPT-4o mini, GPT-4o, o1-mini, o1-preview, and R2D2 models using GPT-3.5 Turbo as a reformulation model. For example, the success rate of this simple attack on GPT-4o increases from 1% using direct requests to 88% using 20 past tense reformulation attempts on harmful requests from JailbreakBench with GPT-4 as a jailbreak judge. Interestingly, we also find that reformulations in the future tense are less effective, suggesting that refusal guardrails tend to consider past historical questions more benign than hypothetical future questions. Moreover, our experiments on fine-tuning GPT-3.5 Turbo show that defending against past reformulations is feasible when past tense examples are explicitly included in the fine-tuning data. Overall, our findings highlight that the widely used alignment techniques -- such as SFT, RLHF, and adversarial training -- employed to align the studied models can be brittle and do not always generalize as intended. We provide code and jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense.

摘要: 拒绝训练被广泛用于防止LLMS产生有害、不受欢迎或非法的输出。我们揭示了当前拒绝训练方法中一个奇怪的概括缺口：简单地用过去时重新表达一个有害的请求(例如，“如何调制燃烧鸡尾酒？”“人们是如何调制燃烧鸡尾酒的？”)通常足以越狱许多最先进的LLM。我们以GPT-3.5 Turbo为改写模型，对Llama-3 8B、Claude-3.5十四行诗、GPT-3.5 Turbo、Gema-2 9B、Phi-3-Mini、GPT-4 o mini、GPT-4 o、o1-mini、o1-PREVIEW和R2D2模型进行了系统的评估。例如，对GPT-4o的这种简单攻击的成功率从使用直接请求的1%增加到使用20次过去时态重组尝试的88%，这些尝试使用GPT-4作为越狱法官的JailBreakB边的有害请求。有趣的是，我们还发现，未来时的重述没有那么有效，这表明拒绝障碍倾向于考虑过去的历史问题，而不是假设的未来问题。此外，我们在微调GPT-3.5Turbo上的实验表明，当微调数据中明确包含过去时态示例时，防御过去的重新公式是可行的。总体而言，我们的发现强调了广泛使用的对齐技术--如SFT、RLHF和对抗性训练--用于对所研究的模型进行对齐可能是脆弱的，并且并不总是像预期的那样泛化。我们在https://github.com/tml-epfl/llm-past-tense.上提供代码和越狱文物



## **30. Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems**

削减开支：基于LLM的多代理系统的经济通信管道 cs.MA

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02506v1) [paper-pdf](http://arxiv.org/pdf/2410.02506v1)

**Authors**: Guibin Zhang, Yanwei Yue, Zhixun Li, Sukwon Yun, Guancheng Wan, Kun Wang, Dawei Cheng, Jeffrey Xu Yu, Tianlong Chen

**Abstract**: Recent advancements in large language model (LLM)-powered agents have shown that collective intelligence can significantly outperform individual capabilities, largely attributed to the meticulously designed inter-agent communication topologies. Though impressive in performance, existing multi-agent pipelines inherently introduce substantial token overhead, as well as increased economic costs, which pose challenges for their large-scale deployments. In response to this challenge, we propose an economical, simple, and robust multi-agent communication framework, termed $\texttt{AgentPrune}$, which can seamlessly integrate into mainstream multi-agent systems and prunes redundant or even malicious communication messages. Technically, $\texttt{AgentPrune}$ is the first to identify and formally define the \textit{communication redundancy} issue present in current LLM-based multi-agent pipelines, and efficiently performs one-shot pruning on the spatial-temporal message-passing graph, yielding a token-economic and high-performing communication topology. Extensive experiments across six benchmarks demonstrate that $\texttt{AgentPrune}$ \textbf{(I)} achieves comparable results as state-of-the-art topologies at merely $\$5.6$ cost compared to their $\$43.7$, \textbf{(II)} integrates seamlessly into existing multi-agent frameworks with $28.1\%\sim72.8\%\downarrow$ token reduction, and \textbf{(III)} successfully defend against two types of agent-based adversarial attacks with $3.5\%\sim10.8\%\uparrow$ performance boost.

摘要: 大型语言模型(LLM)支持的代理的最新进展表明，集体智能可以显著超过个人能力，这在很大程度上要归功于精心设计的代理间通信拓扑。尽管性能令人印象深刻，但现有的多代理管道固有地引入了大量令牌开销，以及增加的经济成本，这对其大规模部署构成了挑战。为了应对这一挑战，我们提出了一个经济、简单、健壮的多智能体通信框架，称为$\exttt{AgentPrune}$，它可以无缝地集成到主流的多智能体系统中，并对冗余甚至恶意的通信消息进行剪枝。从技术上讲，$\exttt{AgentPrune}$是第一个识别和形式化定义当前基于LLM的多代理管道中存在的通信冗余问题的工具，它高效地对时空消息传递图执行一次剪枝，从而产生令牌经济的高性能通信拓扑。在六个基准测试上的广泛实验表明，$\exttt{AgentPrune}$\extbf{(I)}获得了与最先进的拓扑结构相当的结果，与其$\$43.7$相比，只需$5.6$；\extbf{(Ii)}无缝集成到现有的多代理框架中，令牌减少28.1\\sim72.8\%\向下箭头$，并且通过$3.5\%\sim10.8\%\uparrow$性能提升，成功防御两种类型的基于代理的对手攻击。



## **31. Tradeoffs Between Alignment and Helpfulness in Language Models with Representation Engineering**

表示工程语言模型中的一致性和帮助性之间的权衡 cs.CL

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2401.16332v4) [paper-pdf](http://arxiv.org/pdf/2401.16332v4)

**Authors**: Yotam Wolf, Noam Wies, Dorin Shteyman, Binyamin Rothberg, Yoav Levine, Amnon Shashua

**Abstract**: Language model alignment has become an important component of AI safety, allowing safe interactions between humans and language models, by enhancing desired behaviors and inhibiting undesired ones. It is often done by tuning the model or inserting preset aligning prompts. Recently, representation engineering, a method which alters the model's behavior via changing its representations post-training, was shown to be effective in aligning LLMs (Zou et al., 2023a). Representation engineering yields gains in alignment oriented tasks such as resistance to adversarial attacks and reduction of social biases, but was also shown to cause a decrease in the ability of the model to perform basic tasks. In this paper we study the tradeoff between the increase in alignment and decrease in helpfulness of the model. We propose a theoretical framework which provides bounds for these two quantities, and demonstrate their relevance empirically. First, we find that under the conditions of our framework, alignment can be guaranteed with representation engineering, and at the same time that helpfulness is harmed in the process. Second, we show that helpfulness is harmed quadratically with the norm of the representation engineering vector, while the alignment increases linearly with it, indicating a regime in which it is efficient to use representation engineering. We validate our findings empirically, and chart the boundaries to the usefulness of representation engineering for alignment.

摘要: 语言模型对齐已经成为人工智能安全的重要组成部分，通过增强期望的行为和抑制不期望的行为，允许人类和语言模型之间的安全交互。这通常通过调整模型或插入预设对齐提示来完成。最近，表征工程，一种通过在训练后改变模型表征来改变模型行为的方法，被证明在对齐LLM方面是有效的(Zou等人，2023a)。表征工程在对抗对抗性攻击和减少社会偏见等面向对齐的任务中产生收益，但也被证明导致模型执行基本任务的能力下降。在这篇文章中，我们研究了模型的一致性增加和有助性降低之间的权衡。我们提出了一个理论框架，提供了这两个量的界限，并从经验上证明了它们之间的相关性。首先，我们发现，在我们的框架条件下，可以用表示工程来保证对齐，但同时在这个过程中有助性受到了损害。其次，我们证明了有助性与表示工程向量的范数成二次曲线关系，而对齐则随其线性增加，这表明使用表示工程是有效的。我们通过实证验证了我们的发现，并绘制了表征工程对比对的有用性的界限。



## **32. Demonstration Attack against In-Context Learning for Code Intelligence**

针对代码智能的上下文学习的演示攻击 cs.CR

17 pages, 5 figures

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02841v1) [paper-pdf](http://arxiv.org/pdf/2410.02841v1)

**Authors**: Yifei Ge, Weisong Sun, Yihang Lou, Chunrong Fang, Yiran Zhang, Yiming Li, Xiaofang Zhang, Yang Liu, Zhihong Zhao, Zhenyu Chen

**Abstract**: Recent advancements in large language models (LLMs) have revolutionized code intelligence by improving programming productivity and alleviating challenges faced by software developers. To further improve the performance of LLMs on specific code intelligence tasks and reduce training costs, researchers reveal a new capability of LLMs: in-context learning (ICL). ICL allows LLMs to learn from a few demonstrations within a specific context, achieving impressive results without parameter updating. However, the rise of ICL introduces new security vulnerabilities in the code intelligence field. In this paper, we explore a novel security scenario based on the ICL paradigm, where attackers act as third-party ICL agencies and provide users with bad ICL content to mislead LLMs outputs in code intelligence tasks. Our study demonstrates the feasibility and risks of such a scenario, revealing how attackers can leverage malicious demonstrations to construct bad ICL content and induce LLMs to produce incorrect outputs, posing significant threats to system security. We propose a novel method to construct bad ICL content called DICE, which is composed of two stages: Demonstration Selection and Bad ICL Construction, constructing targeted bad ICL content based on the user query and transferable across different query inputs. Ultimately, our findings emphasize the critical importance of securing ICL mechanisms to protect code intelligence systems from adversarial manipulation.

摘要: 大型语言模型(LLM)的最新进展通过提高编程效率和减轻软件开发人员面临的挑战，使代码智能发生了革命性的变化。为了进一步提高LLMS在特定代码智能任务中的性能，降低培训成本，研究人员揭示了LLMS的一种新功能：情境学习(ICL)。ICL允许LLM从特定环境中的几个演示中学习，在不更新参数的情况下取得了令人印象深刻的结果。然而，ICL的兴起在代码情报领域引入了新的安全漏洞。在本文中，我们探索了一种新的基于ICL范式的安全场景，攻击者充当第三方ICL机构，向用户提供不良ICL内容，以在代码情报任务中误导LLMS输出。我们的研究论证了这种情况的可行性和风险，揭示了攻击者如何利用恶意演示来构建不良ICL内容并诱导LLMS产生错误的输出，从而对系统安全构成严重威胁。我们提出了一种构建不良ICL内容的新方法DICE，该方法分为演示选择和不良ICL构建两个阶段，基于用户查询构建具有针对性的不良ICL内容，并可在不同的查询输入之间传输。最后，我们的发现强调了保护ICL机制以保护代码情报系统免受对手操纵的关键重要性。



## **33. Fake It Until You Break It: On the Adversarial Robustness of AI-generated Image Detectors**

伪造直到打破它：人工智能生成图像检测器的对抗鲁棒性 cs.CV

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.01574v2) [paper-pdf](http://arxiv.org/pdf/2410.01574v2)

**Authors**: Sina Mavali, Jonas Ricker, David Pape, Yash Sharma, Asja Fischer, Lea Schönherr

**Abstract**: While generative AI (GenAI) offers countless possibilities for creative and productive tasks, artificially generated media can be misused for fraud, manipulation, scams, misinformation campaigns, and more. To mitigate the risks associated with maliciously generated media, forensic classifiers are employed to identify AI-generated content. However, current forensic classifiers are often not evaluated in practically relevant scenarios, such as the presence of an attacker or when real-world artifacts like social media degradations affect images. In this paper, we evaluate state-of-the-art AI-generated image (AIGI) detectors under different attack scenarios. We demonstrate that forensic classifiers can be effectively attacked in realistic settings, even when the attacker does not have access to the target model and post-processing occurs after the adversarial examples are created, which is standard on social media platforms. These attacks can significantly reduce detection accuracy to the extent that the risks of relying on detectors outweigh their benefits. Finally, we propose a simple defense mechanism to make CLIP-based detectors, which are currently the best-performing detectors, robust against these attacks.

摘要: 虽然生成性人工智能(GenAI)为创造性和富有成效的任务提供了无数可能性，但人工生成的媒体可能被滥用于欺诈、操纵、诈骗、虚假信息活动等。为了降低与恶意生成的媒体相关的风险，使用法医分类器来识别人工智能生成的内容。然而，当前的取证分类器通常不会在实际相关的场景中进行评估，例如攻击者的存在，或者当现实世界中的人工制品(如社交媒体退化)影响图像时。在本文中，我们评估了最新的人工智能生成图像(AIGI)检测器在不同攻击场景下的性能。我们证明了取证分类器在现实环境中可以被有效攻击，即使攻击者没有访问目标模型的权限，并且在创建对抗性示例之后进行后处理，这在社交媒体平台上是标准的。这些攻击可能会大大降低检测的准确性，以至于依赖检测器的风险超过了它们的好处。最后，我们提出了一种简单的防御机制来使目前性能最好的基于CLIP的检测器对这些攻击具有健壮性。



## **34. MOREL: Enhancing Adversarial Robustness through Multi-Objective Representation Learning**

MOREL：通过多目标表示学习增强对抗鲁棒性 cs.LG

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.01697v2) [paper-pdf](http://arxiv.org/pdf/2410.01697v2)

**Authors**: Sedjro Salomon Hotegni, Sebastian Peitz

**Abstract**: Extensive research has shown that deep neural networks (DNNs) are vulnerable to slight adversarial perturbations$-$small changes to the input data that appear insignificant but cause the model to produce drastically different outputs. In addition to augmenting training data with adversarial examples generated from a specific attack method, most of the current defense strategies necessitate modifying the original model architecture components to improve robustness or performing test-time data purification to handle adversarial attacks. In this work, we demonstrate that strong feature representation learning during training can significantly enhance the original model's robustness. We propose MOREL, a multi-objective feature representation learning approach, encouraging classification models to produce similar features for inputs within the same class, despite perturbations. Our training method involves an embedding space where cosine similarity loss and multi-positive contrastive loss are used to align natural and adversarial features from the model encoder and ensure tight clustering. Concurrently, the classifier is motivated to achieve accurate predictions. Through extensive experiments, we demonstrate that our approach significantly enhances the robustness of DNNs against white-box and black-box adversarial attacks, outperforming other methods that similarly require no architectural changes or test-time data purification. Our code is available at https://github.com/salomonhotegni/MOREL

摘要: 广泛的研究表明，深度神经网络(DNN)容易受到输入数据的微小对抗性扰动，这些微小的变化看起来微不足道，但会导致模型产生截然不同的输出。除了使用特定攻击方法生成的对抗性样本来扩充训练数据外，当前的大多数防御策略都需要修改原始模型体系结构组件以提高健壮性，或者执行测试时间数据净化来处理对抗性攻击。在这项工作中，我们证明了在训练过程中的强特征表示学习可以显著增强原始模型的稳健性。我们提出了MOREL，一种多目标特征表示学习方法，鼓励分类模型为同一类内的输入产生相似的特征，尽管存在扰动。我们的训练方法涉及一个嵌入空间，在该空间中使用余弦相似损失和多正对比损失来对齐模型编码器中的自然特征和对抗性特征，并确保紧密的聚类。同时，分类器的动机是实现准确的预测。通过大量的实验，我们证明了我们的方法显著提高了DNN对白盒和黑盒攻击的健壮性，优于其他同样不需要改变体系结构或净化测试时间数据的方法。我们的代码可以在https://github.com/salomonhotegni/MOREL上找到



## **35. AttackBench: Evaluating Gradient-based Attacks for Adversarial Examples**

AttackBench：评估基于攻击的对抗性示例 cs.LG

https://attackbench.github.io

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2404.19460v2) [paper-pdf](http://arxiv.org/pdf/2404.19460v2)

**Authors**: Antonio Emanuele Cinà, Jérôme Rony, Maura Pintor, Luca Demetrio, Ambra Demontis, Battista Biggio, Ismail Ben Ayed, Fabio Roli

**Abstract**: Adversarial examples are typically optimized with gradient-based attacks. While novel attacks are continuously proposed, each is shown to outperform its predecessors using different experimental setups, hyperparameter settings, and number of forward and backward calls to the target models. This provides overly-optimistic and even biased evaluations that may unfairly favor one particular attack over the others. In this work, we aim to overcome these limitations by proposing AttackBench, i.e., the first evaluation framework that enables a fair comparison among different attacks. To this end, we first propose a categorization of gradient-based attacks, identifying their main components and differences. We then introduce our framework, which evaluates their effectiveness and efficiency. We measure these characteristics by (i) defining an optimality metric that quantifies how close an attack is to the optimal solution, and (ii) limiting the number of forward and backward queries to the model, such that all attacks are compared within a given maximum query budget. Our extensive experimental analysis compares more than $100$ attack implementations with a total of over $800$ different configurations against CIFAR-10 and ImageNet models, highlighting that only very few attacks outperform all the competing approaches. Within this analysis, we shed light on several implementation issues that prevent many attacks from finding better solutions or running at all. We release AttackBench as a publicly-available benchmark, aiming to continuously update it to include and evaluate novel gradient-based attacks for optimizing adversarial examples.

摘要: 对抗性示例通常使用基于梯度的攻击进行优化。虽然不断有人提出新的攻击，但通过使用不同的实验设置、超参数设置以及对目标模型的前向和后向调用次数，每个攻击都显示出优于其前辈的性能。这提供了过于乐观甚至有偏见的评估，可能会不公平地偏袒某个特定攻击。在这项工作中，我们的目标是通过提出AttackBtch来克服这些限制，即第一个能够在不同攻击之间进行公平比较的评估框架。为此，我们首先对基于梯度的攻击进行了分类，找出了它们的主要组成部分和区别。然后，我们介绍了我们的框架，它评估了它们的有效性和效率。我们通过(I)定义最优度度量来量化攻击与最优解的距离，以及(Ii)限制对模型的向前和向后查询的数量，以便在给定的最大查询预算内比较所有攻击，来衡量这些特征。我们广泛的实验分析将超过100美元的攻击实施与总计超过800美元的不同配置与CIFAR-10和ImageNet型号进行了比较，强调只有极少数攻击的性能优于所有竞争方法。在这一分析中，我们阐明了几个实现问题，这些问题阻止了许多攻击找到更好的解决方案或根本无法运行。我们发布了一个公开的基准，目的是不断更新它，以包括和评估新的基于梯度的攻击，以优化对手的例子。



## **36. The Role of piracy in quantum proofs**

盗版在量子证明中的作用 quant-ph

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02228v1) [paper-pdf](http://arxiv.org/pdf/2410.02228v1)

**Authors**: Anne Broadbent, Alex B. Grilo, Supartha Podder, Jamie Sikora

**Abstract**: A well-known feature of quantum information is that it cannot, in general, be cloned. Recently, a number of quantum-enabled information-processing tasks have demonstrated various forms of uncloneability; among these forms, piracy is an adversarial model that gives maximal power to the adversary, in controlling both a cloning-type attack, as well as the evaluation/verification stage. Here, we initiate the study of anti-piracy proof systems, which are proof systems that inherently prevent piracy attacks. We define anti-piracy proof systems, demonstrate such a proof system for an oracle problem, and also describe a candidate anti-piracy proof system for NP. We also study quantum proof systems that are cloneable and settle the famous QMA vs. QMA(2) debate in this setting. Lastly, we discuss how one can approach the QMA vs. QCMA question, by studying its cloneable variants.

摘要: 量子信息的一个众所周知的特征是它通常不能被克隆。最近，许多量子使能的信息处理任务已经证明了各种形式的不可克隆性;在这些形式中，盗版是一种对抗模型，它赋予对手最大的权力，以控制克隆型攻击以及评估/验证阶段。在这里，我们启动了反盗版证明系统的研究，这些系统本质上可以防止盗版攻击。我们定义了反盗版证明系统，演示了Oracle问题的这种证明系统，并描述了NP的候选反盗版证明系统。我们还研究可克隆的量子证明系统，并在这种背景下解决著名的QMA与QMA（2）争论。最后，我们讨论如何通过研究其可克隆变体来解决QMA与QCMA问题。



## **37. Semantic-Aware Adversarial Training for Reliable Deep Hashing Retrieval**

用于可靠深度哈希检索的语义感知对抗训练 cs.CV

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2310.14637v2) [paper-pdf](http://arxiv.org/pdf/2310.14637v2)

**Authors**: Xu Yuan, Zheng Zhang, Xunguang Wang, Lin Wu

**Abstract**: Deep hashing has been intensively studied and successfully applied in large-scale image retrieval systems due to its efficiency and effectiveness. Recent studies have recognized that the existence of adversarial examples poses a security threat to deep hashing models, that is, adversarial vulnerability. Notably, it is challenging to efficiently distill reliable semantic representatives for deep hashing to guide adversarial learning, and thereby it hinders the enhancement of adversarial robustness of deep hashing-based retrieval models. Moreover, current researches on adversarial training for deep hashing are hard to be formalized into a unified minimax structure. In this paper, we explore Semantic-Aware Adversarial Training (SAAT) for improving the adversarial robustness of deep hashing models. Specifically, we conceive a discriminative mainstay features learning (DMFL) scheme to construct semantic representatives for guiding adversarial learning in deep hashing. Particularly, our DMFL with the strict theoretical guarantee is adaptively optimized in a discriminative learning manner, where both discriminative and semantic properties are jointly considered. Moreover, adversarial examples are fabricated by maximizing the Hamming distance between the hash codes of adversarial samples and mainstay features, the efficacy of which is validated in the adversarial attack trials. Further, we, for the first time, formulate the formalized adversarial training of deep hashing into a unified minimax optimization under the guidance of the generated mainstay codes. Extensive experiments on benchmark datasets show superb attack performance against the state-of-the-art algorithms, meanwhile, the proposed adversarial training can effectively eliminate adversarial perturbations for trustworthy deep hashing-based retrieval. Our code is available at https://github.com/xandery-geek/SAAT.

摘要: 深度散列算法以其高效、高效的特点在大规模图像检索系统中得到了广泛的研究和成功的应用。最近的研究已经认识到，对抗性例子的存在对深度哈希模型构成了安全威胁，即对抗性漏洞。值得注意的是，有效地提取可靠的语义代表用于深度散列以指导对抗性学习是具有挑战性的，从而阻碍了基于深度散列的检索模型对抗性健壮性的增强。此外，目前针对深度散列的对抗性训练的研究很难被形式化成一个统一的极大极小结构。本文探讨了语义感知对抗训练(SAAT)来提高深度哈希模型的对抗健壮性。具体地说，我们设想了一种区分主干特征学习(DMFL)方案来构建语义表示，以指导深度哈希中的对抗性学习。特别是，我们的DMFL在严格的理论保证下，以区分学习的方式进行了自适应优化，同时考虑了区分属性和语义属性。此外，通过最大化对抗性样本的哈希码与主流特征之间的汉明距离来构造对抗性样本，并在对抗性攻击试验中验证了该方法的有效性。在生成的主干代码的指导下，首次将深度散列的形式化对抗性训练转化为统一的极大极小优化问题。在基准数据集上的大量实验表明，该算法对现有算法具有良好的攻击性能，同时，本文提出的对抗性训练能够有效地消除对抗性扰动，实现基于深度散列的可信检索。我们的代码可以在https://github.com/xandery-geek/SAAT.上找到



## **38. Securing Cloud File Systems with Trusted Execution**

通过可信执行保护云文件系统 cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2305.18639v3) [paper-pdf](http://arxiv.org/pdf/2305.18639v3)

**Authors**: Quinn Burke, Yohan Beugin, Blaine Hoak, Rachel King, Eric Pauley, Ryan Sheatsley, Mingli Yu, Ting He, Thomas La Porta, Patrick McDaniel

**Abstract**: Cloud file systems offer organizations a scalable and reliable file storage solution. However, cloud file systems have become prime targets for adversaries, and traditional designs are not equipped to protect organizations against the myriad of attacks that may be initiated by a malicious cloud provider, co-tenant, or end-client. Recently proposed designs leveraging cryptographic techniques and trusted execution environments (TEEs) still force organizations to make undesirable trade-offs, consequently leading to either security, functional, or performance limitations. In this paper, we introduce BFS, a cloud file system that leverages the security capabilities provided by TEEs to bootstrap new security protocols that deliver strong security guarantees, high-performance, and a transparent POSIX-like interface to clients. BFS delivers stronger security guarantees and up to a 2.5X speedup over a state-of-the-art secure file system. Moreover, compared to the industry standard NFS, BFS achieves up to 2.2X speedups across micro-benchmarks and incurs <1X overhead for most macro-benchmark workloads. BFS demonstrates a holistic cloud file system design that does not sacrifice an organizations' security yet can embrace all of the functional and performance advantages of outsourcing.

摘要: 云文件系统为组织提供了可扩展且可靠的文件存储解决方案。然而，云文件系统已成为对手的主要目标，传统设计无法保护组织免受恶意云提供商、联合租户或终端客户端可能发起的无数攻击。最近提出的利用加密技术和可信执行环境(TEE)的设计仍然迫使组织做出不希望看到的权衡，从而导致安全、功能或性能限制。在本文中，我们介绍了BFS，一个云文件系统，它利用TES提供的安全能力来引导新的安全协议，为客户端提供强大的安全保证、高性能和透明的POSIX类接口。与最先进的安全文件系统相比，BFS提供了更强大的安全保证和高达2.5倍的加速。此外，与行业标准的NFS相比，BFS在微观基准测试中实现了高达2.2倍的加速，而对于大多数宏观基准测试工作负载，其开销不到1倍。BFS展示了一种整体的云文件系统设计，该设计既不会牺牲组织的安全性，又可以包含外包的所有功能和性能优势。



## **39. Impact of White-Box Adversarial Attacks on Convolutional Neural Networks**

白盒对抗攻击对卷积神经网络的影响 cs.CR

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.02043v1) [paper-pdf](http://arxiv.org/pdf/2410.02043v1)

**Authors**: Rakesh Podder, Sudipto Ghosh

**Abstract**: Autonomous vehicle navigation and healthcare diagnostics are among the many fields where the reliability and security of machine learning models for image data are critical. We conduct a comprehensive investigation into the susceptibility of Convolutional Neural Networks (CNNs), which are widely used for image data, to white-box adversarial attacks. We investigate the effects of various sophisticated attacks -- Fast Gradient Sign Method, Basic Iterative Method, Jacobian-based Saliency Map Attack, Carlini & Wagner, Projected Gradient Descent, and DeepFool -- on CNN performance metrics, (e.g., loss, accuracy), the differential efficacy of adversarial techniques in increasing error rates, the relationship between perceived image quality metrics (e.g., ERGAS, PSNR, SSIM, and SAM) and classification performance, and the comparative effectiveness of iterative versus single-step attacks. Using the MNIST, CIFAR-10, CIFAR-100, and Fashio_MNIST datasets, we explore the effect of different attacks on the CNNs performance metrics by varying the hyperparameters of CNNs. Our study provides insights into the robustness of CNNs against adversarial threats, pinpoints vulnerabilities, and underscores the urgent need for developing robust defense mechanisms to protect CNNs and ensuring their trustworthy deployment in real-world scenarios.

摘要: 在许多领域中，图像数据的机器学习模型的可靠性和安全性至关重要，自动车辆导航和医疗诊断就是其中之一。本文对广泛用于图像数据的卷积神经网络(CNN)对白盒攻击的敏感性进行了全面的研究。我们研究了各种复杂的攻击--快速梯度符号方法、基本迭代方法、基于雅可比的显著图攻击、Carlini&Wagner、投影梯度下降和DeepFool--对CNN性能指标(例如损失、准确度)的影响、对抗性技术在提高错误率方面的不同有效性、感知图像质量指标(例如Ergas、PSNR、SSIM和SAM)与分类性能之间的关系，以及迭代攻击与单步攻击的比较有效性。使用MNIST、CIFAR-10、CIFAR-100和THAMO_MNIST数据集，通过改变CNN的超参数，我们探索了不同攻击对CNN性能指标的影响。我们的研究提供了对CNN对抗对手威胁的稳健性的见解，准确地指出了漏洞，并强调了迫切需要开发稳健的防御机制来保护CNN，并确保其在现实世界场景中的可信部署。



## **40. EAB-FL: Exacerbating Algorithmic Bias through Model Poisoning Attacks in Federated Learning**

EAB-FL：通过联邦学习中的模型中毒攻击加剧数学偏见 cs.LG

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.02042v1) [paper-pdf](http://arxiv.org/pdf/2410.02042v1)

**Authors**: Syed Irfan Ali Meerza, Jian Liu

**Abstract**: Federated Learning (FL) is a technique that allows multiple parties to train a shared model collaboratively without disclosing their private data. It has become increasingly popular due to its distinct privacy advantages. However, FL models can suffer from biases against certain demographic groups (e.g., racial and gender groups) due to the heterogeneity of data and party selection. Researchers have proposed various strategies for characterizing the group fairness of FL algorithms to address this issue. However, the effectiveness of these strategies in the face of deliberate adversarial attacks has not been fully explored. Although existing studies have revealed various threats (e.g., model poisoning attacks) against FL systems caused by malicious participants, their primary aim is to decrease model accuracy, while the potential of leveraging poisonous model updates to exacerbate model unfairness remains unexplored. In this paper, we propose a new type of model poisoning attack, EAB-FL, with a focus on exacerbating group unfairness while maintaining a good level of model utility. Extensive experiments on three datasets demonstrate the effectiveness and efficiency of our attack, even with state-of-the-art fairness optimization algorithms and secure aggregation rules employed.

摘要: 联合学习(FL)是一种允许多方协作训练共享模型而不披露他们的私人数据的技术。由于其独特的隐私优势，它变得越来越受欢迎。然而，由于数据和政党选择的异质性，FL模型可能会受到对某些人口统计群体(例如，种族和性别群体)的偏见。为了解决这个问题，研究人员提出了不同的策略来刻画FL算法的群体公平性。然而，面对蓄意的敌意攻击，这些战略的有效性尚未得到充分探讨。虽然现有的研究已经揭示了恶意参与者对FL系统造成的各种威胁(例如，模型中毒攻击)，但它们的主要目的是降低模型的准确性，而利用有毒模型更新来加剧模型不公平的可能性仍未被探索。在本文中，我们提出了一种新型的模型中毒攻击，EAB-FL，其重点是在保持良好的模型效用的同时加剧群体不公平。在三个数据集上的大量实验证明了我们的攻击的有效性和效率，即使使用了最先进的公平优化算法和安全聚合规则。



## **41. CLIP-Guided Generative Networks for Transferable Targeted Adversarial Attacks**

CLIP引导的生成网络用于可转移有针对性的对抗攻击 cs.CV

ECCV 2024

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2407.10179v3) [paper-pdf](http://arxiv.org/pdf/2407.10179v3)

**Authors**: Hao Fang, Jiawei Kong, Bin Chen, Tao Dai, Hao Wu, Shu-Tao Xia

**Abstract**: Transferable targeted adversarial attacks aim to mislead models into outputting adversary-specified predictions in black-box scenarios. Recent studies have introduced \textit{single-target} generative attacks that train a generator for each target class to generate highly transferable perturbations, resulting in substantial computational overhead when handling multiple classes. \textit{Multi-target} attacks address this by training only one class-conditional generator for multiple classes. However, the generator simply uses class labels as conditions, failing to leverage the rich semantic information of the target class. To this end, we design a \textbf{C}LIP-guided \textbf{G}enerative \textbf{N}etwork with \textbf{C}ross-attention modules (CGNC) to enhance multi-target attacks by incorporating textual knowledge of CLIP into the generator. Extensive experiments demonstrate that CGNC yields significant improvements over previous multi-target generative attacks, e.g., a 21.46\% improvement in success rate from ResNet-152 to DenseNet-121. Moreover, we propose a masked fine-tuning mechanism to further strengthen our method in attacking a single class, which surpasses existing single-target methods.

摘要: 可转移的目标对抗性攻击旨在误导模型，使其在黑盒场景中输出对手指定的预测。最近的研究引入了生成性攻击，这种攻击为每个目标类训练一个生成器来生成高度可传递的扰动，导致在处理多个类时产生大量的计算开销。\textit{多目标}攻击通过仅训练多个类的一个类条件生成器来解决此问题。然而，生成器简单地使用类标签作为条件，没有利用目标类的丰富语义信息。为此，我们设计了一个唇形引导的生成模块(CGNC)，通过在生成器中加入剪辑文本知识来增强多目标攻击。大量的实验表明，CGNC比以前的多目标生成性攻击有显著的改进，例如，成功率从ResNet-152提高到DenseNet-121，提高了21.46%.此外，我们还提出了一种屏蔽微调机制，进一步加强了我们的攻击单一类的方法，超越了现有的单目标攻击方法。



## **42. Social Media Authentication and Combating Deepfakes using Semi-fragile Invisible Image Watermarking**

使用半脆弱不可见图像水印进行社交媒体验证和打击Deepfakes cs.CV

ACM Transactions (Digital Threats: Research and Practice)

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.01906v1) [paper-pdf](http://arxiv.org/pdf/2410.01906v1)

**Authors**: Aakash Varma Nadimpalli, Ajita Rattani

**Abstract**: With the significant advances in deep generative models for image and video synthesis, Deepfakes and manipulated media have raised severe societal concerns. Conventional machine learning classifiers for deepfake detection often fail to cope with evolving deepfake generation technology and are susceptible to adversarial attacks. Alternatively, invisible image watermarking is being researched as a proactive defense technique that allows media authentication by verifying an invisible secret message embedded in the image pixels. A handful of invisible image watermarking techniques introduced for media authentication have proven vulnerable to basic image processing operations and watermark removal attacks. In response, we have proposed a semi-fragile image watermarking technique that embeds an invisible secret message into real images for media authentication. Our proposed watermarking framework is designed to be fragile to facial manipulations or tampering while being robust to benign image-processing operations and watermark removal attacks. This is facilitated through a unique architecture of our proposed technique consisting of critic and adversarial networks that enforce high image quality and resiliency to watermark removal efforts, respectively, along with the backbone encoder-decoder and the discriminator networks. Thorough experimental investigations on SOTA facial Deepfake datasets demonstrate that our proposed model can embed a $64$-bit secret as an imperceptible image watermark that can be recovered with a high-bit recovery accuracy when benign image processing operations are applied while being non-recoverable when unseen Deepfake manipulations are applied. In addition, our proposed watermarking technique demonstrates high resilience to several white-box and black-box watermark removal attacks. Thus, obtaining state-of-the-art performance.

摘要: 随着图像和视频合成的深度生成模型的重大进步，Deepfake和被操纵的媒体已经引起了严重的社会关注。传统的用于深度伪检测的机器学习分类器往往无法应对不断发展的深度伪生成技术，并且容易受到对手攻击。或者，隐形图像水印正在作为一种主动防御技术进行研究，该技术允许通过验证嵌入在图像像素中的不可见秘密消息来进行媒体认证。一些用于媒体认证的不可见图像水印技术已被证明容易受到基本图像处理操作和水印移除攻击。对此，我们提出了一种半脆弱图像水印技术，将不可见的秘密信息嵌入到真实图像中进行媒体认证。我们提出的水印框架被设计成对面部操作或篡改是脆弱的，而对良性图像处理操作和水印去除攻击是健壮的。这是通过我们建议的技术的独特体系结构来促进的，该体系结构包括分别执行高图像质量和对水印去除努力的弹性的批评性和对抗性网络，以及主干编解码器和鉴别器网络。在Sota人脸深伪数据集上的实验研究表明，该模型可以嵌入一个$$比特的秘密作为不可感知的图像水印，当应用良性图像处理操作时可以高比特恢复精度，而当应用不可见的深伪操作时是不可恢复的。此外，我们提出的水印技术对几种白盒和黑盒水印去除攻击都表现出了很高的抗攻击能力。从而获得最先进的性能。



## **43. MeanSparse: Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification**

MeanSparse：通过以均值为中心的特征稀疏化来增强训练后的鲁棒性 cs.CV

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2406.05927v2) [paper-pdf](http://arxiv.org/pdf/2406.05927v2)

**Authors**: Sajjad Amini, Mohammadreza Teymoorianfard, Shiqing Ma, Amir Houmansadr

**Abstract**: We present a simple yet effective method to improve the robustness of both Convolutional and attention-based Neural Networks against adversarial examples by post-processing an adversarially trained model. Our technique, MeanSparse, cascades the activation functions of a trained model with novel operators that sparsify mean-centered feature vectors. This is equivalent to reducing feature variations around the mean, and we show that such reduced variations merely affect the model's utility, yet they strongly attenuate the adversarial perturbations and decrease the attacker's success rate. Our experiments show that, when applied to the top models in the RobustBench leaderboard, MeanSparse achieves a new robustness record of 75.28% (from 73.71%), 44.78% (from 42.67%) and 62.12% (from 59.56%) on CIFAR-10, CIFAR-100 and ImageNet, respectively, in terms of AutoAttack accuracy. Code is available at https://github.com/SPIN-UMass/MeanSparse

摘要: 我们提出了一种简单而有效的方法，通过对对抗训练的模型进行后处理，提高卷积和基于注意力的神经网络针对对抗示例的鲁棒性。我们的技术MeanSparse通过新颖的运算符级联经过训练的模型的激活函数，这些运算符稀疏化以均值为中心的特征载体。这相当于减少均值附近的特征变化，我们表明，这种减少的变化只会影响模型的效用，但它们会强烈削弱对抗性扰动并降低攻击者的成功率。我们的实验表明，当应用于RobustBench排行榜上的顶级模型时，MeanSparse在CIFAR-10、CIFAR-100和ImageNet上分别实现了75.28%（从73.71%开始）、44.78%（从42.67%开始）和62.12%（从59.56%开始）的新稳健性记录。代码可访问https://github.com/SPIN-UMass/MeanSparse



## **44. KeyVisor -- A Lightweight ISA Extension for Protected Key Handles with CPU-enforced Usage Policies**

KeyVisor --一个轻量级ISA扩展，用于受保护的密钥手柄，具有MCU强制的使用策略 cs.CR

preprint

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.01777v1) [paper-pdf](http://arxiv.org/pdf/2410.01777v1)

**Authors**: Fabian Schwarz, Jan Philipp Thoma, Christian Rossow, Tim Güneysu

**Abstract**: The confidentiality of cryptographic keys is essential for the security of protection schemes used for communication, file encryption, and outsourced computation. Beyond cryptanalytic attacks, adversaries can steal keys from memory via software exploits or side channels, enabling them to, e.g., tamper with secrets or impersonate key owners. Therefore, existing defenses protect keys in dedicated devices or isolated memory, or store them only in encrypted form. However, these designs often provide unfavorable tradeoffs, sacrificing performance, fine-grained access control, or deployability.   In this paper, we present KeyVisor, a lightweight ISA extension that securely offloads the handling of cryptographic keys to the CPU. KeyVisor provides CPU instructions that enable applications to request protected key handles and perform AEAD cipher operations on them. The underlying keys are accessible only by KeyVisor, and thus never leak to memory. KeyVisor's direct CPU integration enables fast crypto operations and hardware-enforced key usage restrictions, e.g., keys usable only for de-/encryption, with a limited lifetime, or with a process binding. Furthermore, privileged software, e.g., the monitor firmware of TEEs, can revoke keys or bind them to a specific process/TEE. We implement KeyVisor for RISC-V based on Rocket Chip, evaluate its performance, and demonstrate real-world use cases, including key-value databases, automotive feature licensing, and a read-only network middlebox.

摘要: 加密密钥的机密性对于用于通信、文件加密和外包计算的保护方案的安全性至关重要。除了密码分析攻击之外，攻击者还可以通过软件漏洞攻击或旁路从内存中窃取密钥，使他们能够例如篡改机密或冒充密钥所有者。因此，现有的防御措施保护专用设备或隔离存储器中的密钥，或仅以加密形式存储它们。然而，这些设计经常提供不利的折衷，牺牲性能、细粒度访问控制或可部署性。在本文中，我们介绍了KeyVisor，这是一个轻量级ISA扩展，可以安全地将加密密钥的处理卸载到CPU。KeyVisor提供了CPU指令，使应用程序能够请求受保护的密钥句柄并对其执行AEAD密码操作。基础键只能由KeyVisor访问，因此永远不会泄漏到内存。KeyVisor的直接CPU集成实现了快速加密操作和硬件强制的密钥使用限制，例如，仅可用于解密/加密的密钥、有限寿命的密钥或具有进程绑定的密钥。此外，特权软件，例如TEE的监控固件，可以撤销密钥或将它们绑定到特定进程/TEE。我们基于Rocket Chip为RISC-V实现了KeyVisor，对其性能进行了评估，并展示了真实世界的用例，包括键值数据库、汽车特征许可和只读网络中间盒。



## **45. Towards Understanding the Robustness of Diffusion-Based Purification: A Stochastic Perspective**

了解基于扩散的净化的稳健性：随机视角 cs.CV

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2404.14309v2) [paper-pdf](http://arxiv.org/pdf/2404.14309v2)

**Authors**: Yiming Liu, Kezhao Liu, Yao Xiao, Ziyi Dong, Xiaogang Xu, Pengxu Wei, Liang Lin

**Abstract**: Diffusion-Based Purification (DBP) has emerged as an effective defense mechanism against adversarial attacks. The efficacy of DBP has been attributed to the forward diffusion process, which narrows the distribution gap between clean and adversarial images through the addition of Gaussian noise. Although this explanation has some theoretical support, the significance of its contribution to robustness remains unclear. In this paper, we argue that the inherent stochasticity in the DBP process is the primary driver of its robustness. To explore this, we introduce a novel Deterministic White-Box (DW-box) evaluation protocol to assess robustness in the absence of stochasticity and to analyze the attack trajectories and loss landscapes. Our findings suggest that DBP models primarily leverage stochasticity to evade effective attack directions, and their ability to purify adversarial perturbations can be weak. To further enhance the robustness of DBP models, we introduce Adversarial Denoising Diffusion Training (ADDT), which incorporates classifier-guided adversarial perturbations into diffusion training, thereby strengthening the DBP models' ability to purify adversarial perturbations. Additionally, we propose Rank-Based Gaussian Mapping (RBGM) to make perturbations more compatible with diffusion models. Experimental results validate the effectiveness of ADDT. In conclusion, our study suggests that future research on DBP can benefit from the perspective of decoupling the stochasticity-based and purification-based robustness.

摘要: 基于扩散的净化技术(DBP)已成为抵抗敌意攻击的一种有效防御机制。DBP的有效性归因于前向扩散过程，该过程通过添加高斯噪声缩小了干净图像和敌对图像之间的分布差距。尽管这一解释有一定的理论支持，但它对稳健性的贡献的意义仍然不清楚。在本文中，我们认为DBP过程固有的随机性是其稳健性的主要驱动因素。为了探索这一点，我们引入了一种新的确定性白盒(DW-box)评估协议来评估在缺乏随机性的情况下的稳健性，并分析攻击轨迹和损失情况。我们的发现表明，DBP模型主要利用随机性来规避有效的攻击方向，并且它们净化对手扰动的能力可能较弱。为了进一步增强DBP模型的稳健性，我们引入了对抗性去噪扩散训练(ADDT)，它将分类器引导的对抗性扰动融入扩散训练中，从而增强了DBP模型净化对抗性扰动的能力。此外，我们还提出了基于秩高斯映射(RBGM)，以使扰动与扩散模型更兼容。实验结果验证了该算法的有效性。总之，我们的研究表明，DBP的未来研究可以从解耦基于随机性的稳健性和基于净化的稳健性的角度受益。



## **46. On Scaling LT-Coded Blockchains in Heterogeneous Networks and their Vulnerabilities to DoS Threats**

关于在异类网络中扩展LT编码区块链及其对拒绝服务威胁的脆弱性 cs.IT

Extended version of the results presented at IEEE ICC 2024

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2402.05620v2) [paper-pdf](http://arxiv.org/pdf/2402.05620v2)

**Authors**: Harikrishnan K., J. Harshan, Anwitaman Datta

**Abstract**: Coded blockchains have acquired prominence as a promising solution to reduce storage costs and facilitate scalability. Within this class, Luby Transform (LT) coded blockchains are an appealing choice for scalability owing to the availability of a wide range of low-complexity decoders. In the first part of this work, we identify that traditional LT decoders like Belief Propagation and On-the-Fly Gaussian Elimination may not be optimal for heterogeneous networks with nodes that have varying computational and download capabilities. To address this, we introduce a family of hybrid decoders for LT codes and propose optimal operating regimes for them to recover the blockchain at the lowest decoding cost. While LT coded blockchain architecture has been studied from the aspects of storage savings and scalability, not much is known in terms of its security vulnerabilities. Pointing at this research gap, in the second part, we present novel denial-of-service threats on LT coded blockchains that target nodes with specific decoding capabilities, preventing them from joining the network. Our proposed threats are non-oblivious in nature, wherein adversaries gain access to the archived blocks, and choose to execute their attack on a subset of them based on underlying coding scheme. We show that our optimized threats can achieve the same level of damage as that of blind attacks, however, with limited amount of resources. Overall, this is the first work of its kind that opens up new questions on designing coded blockchains to jointly provide storage savings, scalability and also resilience to optimized threats.

摘要: 编码区块链作为一种降低存储成本和促进可扩展性的有前途的解决方案而获得了突出的地位。在这一类中，Luby变换(LT)编码的区块链是可扩展性的一个有吸引力的选择，因为有广泛的低复杂度解码器可用。在这项工作的第一部分，我们发现传统的LT解码器，如信任传播和即时高斯消除，对于具有不同计算和下载能力的节点的异类网络可能不是最优的。为了解决这一问题，我们引入了一系列用于LT码的混合译码，并提出了它们以最低译码成本恢复区块链的最佳运行机制。虽然LT编码的区块链架构已经从存储节省和可扩展性方面进行了研究，但在其安全漏洞方面却知之甚少。针对这一研究空白，在第二部分中，我们提出了一种针对具有特定解码能力的节点的LT编码区块链上的新型拒绝服务威胁，阻止它们加入网络。我们提出的威胁在本质上是不可忽视的，其中攻击者获得对归档块的访问权限，并选择基于底层编码方案对其中的子集执行攻击。我们表明，我们的优化威胁可以达到与盲目攻击相同的损害水平，然而，使用有限的资源。总体而言，这是此类工作中的第一项，它打开了设计编码区块链的新问题，以共同提供存储节省、可扩展性以及对优化威胁的弹性。



## **47. On Using Certified Training towards Empirical Robustness**

关于使用认证培训来提高经验稳健性 cs.LG

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.01617v1) [paper-pdf](http://arxiv.org/pdf/2410.01617v1)

**Authors**: Alessandro De Palma, Serge Durand, Zakaria Chihani, François Terrier, Caterina Urban

**Abstract**: Adversarial training is arguably the most popular way to provide empirical robustness against specific adversarial examples. While variants based on multi-step attacks incur significant computational overhead, single-step variants are vulnerable to a failure mode known as catastrophic overfitting, which hinders their practical utility for large perturbations. A parallel line of work, certified training, has focused on producing networks amenable to formal guarantees of robustness against any possible attack. However, the wide gap between the best-performing empirical and certified defenses has severely limited the applicability of the latter. Inspired by recent developments in certified training, which rely on a combination of adversarial attacks with network over-approximations, and by the connections between local linearity and catastrophic overfitting, we present experimental evidence on the practical utility and limitations of using certified training towards empirical robustness. We show that, when tuned for the purpose, a recent certified training algorithm can prevent catastrophic overfitting on single-step attacks, and that it can bridge the gap to multi-step baselines under appropriate experimental settings. Finally, we present a novel regularizer for network over-approximations that can achieve similar effects while markedly reducing runtime.

摘要: 对抗性训练可以说是针对特定对抗性例子提供经验稳健性的最受欢迎的方式。虽然基于多步攻击的变体会导致巨大的计算开销，但单步变体容易受到一种称为灾难性过拟合的故障模式的影响，这阻碍了它们在处理大扰动时的实用价值。另一项并行不悖的工作是认证培训，重点是生产能够遵守针对任何可能攻击的健壮性的正式保证的网络。然而，表现最好的经验辩护和认证辩护之间的巨大差距严重限制了后者的适用性。受认证训练的最新发展的启发，这些发展依赖于对抗性攻击与网络过逼近的组合，以及局部线性和灾难性过拟合之间的联系，我们提供了实验证据，证明使用认证训练对于经验稳健性的实际效用和局限性。我们证明，当为此目的进行调整时，最近经过认证的训练算法可以防止单步攻击的灾难性过拟合，并且在适当的实验设置下，它可以将差距弥合到多步基线。最后，我们提出了一种新的网络过逼近正则化方法，它可以在显著减少运行时间的情况下达到类似的效果。



## **48. Automated Red Teaming with GOAT: the Generative Offensive Agent Tester**

自动Red与GOAT合作：生成式进攻代理测试器 cs.LG

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.01606v1) [paper-pdf](http://arxiv.org/pdf/2410.01606v1)

**Authors**: Maya Pavlova, Erik Brinkman, Krithika Iyer, Vitor Albiero, Joanna Bitton, Hailey Nguyen, Joe Li, Cristian Canton Ferrer, Ivan Evtimov, Aaron Grattafiori

**Abstract**: Red teaming assesses how large language models (LLMs) can produce content that violates norms, policies, and rules set during their safety training. However, most existing automated methods in the literature are not representative of the way humans tend to interact with AI models. Common users of AI models may not have advanced knowledge of adversarial machine learning methods or access to model internals, and they do not spend a lot of time crafting a single highly effective adversarial prompt. Instead, they are likely to make use of techniques commonly shared online and exploit the multiturn conversational nature of LLMs. While manual testing addresses this gap, it is an inefficient and often expensive process. To address these limitations, we introduce the Generative Offensive Agent Tester (GOAT), an automated agentic red teaming system that simulates plain language adversarial conversations while leveraging multiple adversarial prompting techniques to identify vulnerabilities in LLMs. We instantiate GOAT with 7 red teaming attacks by prompting a general-purpose model in a way that encourages reasoning through the choices of methods available, the current target model's response, and the next steps. Our approach is designed to be extensible and efficient, allowing human testers to focus on exploring new areas of risk while automation covers the scaled adversarial stress-testing of known risk territory. We present the design and evaluation of GOAT, demonstrating its effectiveness in identifying vulnerabilities in state-of-the-art LLMs, with an ASR@10 of 97% against Llama 3.1 and 88% against GPT-4 on the JailbreakBench dataset.

摘要: 红色团队评估大型语言模型(LLM)在多大程度上可以产生违反其安全培训期间设定的规范、政策和规则的内容。然而，文献中的大多数现有自动化方法都不能代表人类与人工智能模型交互的方式。AI模型的普通用户可能没有对抗性机器学习方法的高级知识，也没有访问模型内部的权限，他们也不会花费大量时间来制作单个高效的对抗性提示。取而代之的是，他们可能会利用在线共享的常见技术，并利用LLMS的多轮对话性质。虽然手动测试弥补了这一差距，但它是一个低效且往往昂贵的过程。为了解决这些局限性，我们引入了生成性进攻代理Tester(山羊)，这是一个自动化的代理红色团队系统，它模拟普通语言的对抗性对话，同时利用多个对抗性提示技术来识别LLMS中的漏洞。我们用7个红色团队攻击实例化山羊，通过提示通用模型的方式，通过选择可用方法、当前目标模型的响应和下一步来鼓励推理。我们的方法被设计为可扩展和高效的，允许人工测试人员专注于探索新的风险领域，而自动化覆盖了已知风险领域的大规模对抗性压力测试。我们给出了山羊的设计和评估，展示了它在识别最新LLMS漏洞方面的有效性，在JailBreakB边数据集上，ASR@10对Llama 3.1的ASR为97%，对GPT-4的ASR为88%。



## **49. $σ$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples**

$Sigma $-zero：$\ell_0 $-norm的基于对象的优化对抗性示例 cs.LG

Code available at  https://github.com/Cinofix/sigma-zero-adversarial-attack

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2402.01879v2) [paper-pdf](http://arxiv.org/pdf/2402.01879v2)

**Authors**: Antonio Emanuele Cinà, Francesco Villani, Maura Pintor, Lea Schönherr, Battista Biggio, Marcello Pelillo

**Abstract**: Evaluating the adversarial robustness of deep networks to gradient-based attacks is challenging. While most attacks consider $\ell_2$- and $\ell_\infty$-norm constraints to craft input perturbations, only a few investigate sparse $\ell_1$- and $\ell_0$-norm attacks. In particular, $\ell_0$-norm attacks remain the least studied due to the inherent complexity of optimizing over a non-convex and non-differentiable constraint. However, evaluating adversarial robustness under these attacks could reveal weaknesses otherwise left untested with more conventional $\ell_2$- and $\ell_\infty$-norm attacks. In this work, we propose a novel $\ell_0$-norm attack, called $\sigma$-zero, which leverages a differentiable approximation of the $\ell_0$ norm to facilitate gradient-based optimization, and an adaptive projection operator to dynamically adjust the trade-off between loss minimization and perturbation sparsity. Extensive evaluations using MNIST, CIFAR10, and ImageNet datasets, involving robust and non-robust models, show that $\sigma$-zero finds minimum $\ell_0$-norm adversarial examples without requiring any time-consuming hyperparameter tuning, and that it outperforms all competing sparse attacks in terms of success rate, perturbation size, and efficiency.

摘要: 评估深度网络对抗基于梯度的攻击的健壮性是具有挑战性的。虽然大多数攻击考虑$\ell_2$-和$\ell_\inty$-范数约束来手工创建输入扰动，但只有少数攻击研究稀疏的$\ell_1$-和$\ell_0$-范数攻击。特别是，由于在非凸和不可微约束上进行优化的固有复杂性，$\ell_0$-范数攻击仍然是研究最少的。然而，在这些攻击下评估对手的健壮性可能会揭示出在更常规的$\ell_2$-和$\ell_\inty$-范数攻击中未被测试的弱点。在这项工作中，我们提出了一种新的$\ell_0$-范数攻击，称为$\sigma$-零，它利用$\ell_0$范数的可微近似来促进基于梯度的优化，并提出了一种自适应投影算子来动态调整损失最小化和扰动稀疏性之间的权衡。使用MNIST、CIFAR10和ImageNet数据集进行的广泛评估，包括健壮和非健壮模型，表明$\sigma$-Zero在不需要任何耗时的超参数调整的情况下找到最小$\ell_0$-范数对手示例，并且在成功率、扰动大小和效率方面优于所有竞争的稀疏攻击。



## **50. Signal Adversarial Examples Generation for Signal Detection Network via White-Box Attack**

通过白盒攻击生成信号检测网络的信号对抗示例 cs.CV

18 pages, 6 figures, submitted to Mobile Networks and Applications

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.01393v1) [paper-pdf](http://arxiv.org/pdf/2410.01393v1)

**Authors**: Dongyang Li, Linyuan Wang, Guangwei Xiong, Bin Yan, Dekui Ma, Jinxian Peng

**Abstract**: With the development and application of deep learning in signal detection tasks, the vulnerability of neural networks to adversarial attacks has also become a security threat to signal detection networks. This paper defines a signal adversarial examples generation model for signal detection network from the perspective of adding perturbations to the signal. The model uses the inequality relationship of L2-norm between time domain and time-frequency domain to constrain the energy of signal perturbations. Building upon this model, we propose a method for generating signal adversarial examples utilizing gradient-based attacks and Short-Time Fourier Transform. The experimental results show that under the constraint of signal perturbation energy ratio less than 3%, our adversarial attack resulted in a 28.1% reduction in the mean Average Precision (mAP), a 24.7% reduction in recall, and a 30.4% reduction in precision of the signal detection network. Compared to random noise perturbation of equivalent intensity, our adversarial attack demonstrates a significant attack effect.

摘要: 随着深度学习在信号检测任务中的发展和应用，神经网络对敌意攻击的脆弱性也成为信号检测网络的安全威胁。从信号中加入扰动的角度出发，定义了信号检测网络的信号对抗性实例生成模型。该模型利用时频域和时频域之间的L2范数不等关系来约束信号扰动的能量。在此模型的基础上，我们提出了一种利用基于梯度的攻击和短时傅立叶变换生成信号对抗性样本的方法。实验结果表明，在信号扰动能量比小于3%的约束下，我们的对抗性攻击导致信号检测网络的平均准确率(MAP)下降了28.1%，召回率下降了24.7%，准确率下降了30.4%。与同等强度的随机噪声扰动相比，我们的对抗性攻击具有显著的攻击效果。



