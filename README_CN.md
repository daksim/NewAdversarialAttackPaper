# Latest Adversarial Attack Papers
**update at 2024-08-22 17:18:42**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Competence-Based Analysis of Language Models**

基于能力的语言模型分析 cs.CL

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2303.00333v4) [paper-pdf](http://arxiv.org/pdf/2303.00333v4)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent successes of large, pretrained neural language models (LLMs), comparatively little is known about the representations of linguistic structure they learn during pretraining, which can lead to unexpected behaviors in response to prompt variation or distribution shift. To better understand these models and behaviors, we introduce a general model analysis framework to study LLMs with respect to their representation and use of human-interpretable linguistic properties. Our framework, CALM (Competence-based Analysis of Language Models), is designed to investigate LLM competence in the context of specific tasks by intervening on models' internal representations of different linguistic properties using causal probing, and measuring models' alignment under these interventions with a given ground-truth causal model of the task. We also develop a new approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than prior techniques. Finally, we carry out a case study of CALM using these interventions to analyze and compare LLM competence across a variety of lexical inference tasks, showing that CALM can be used to explain and predict behaviors across these tasks.

摘要: 尽管最近大型的预训练神经语言模型(LLM)取得了成功，但人们对它们在预训练中学习的语言结构的表征知之甚少，这可能会导致对迅速变化或分布变化的意外行为。为了更好地理解这些模型和行为，我们引入了一个通用的模型分析框架，从它们对人类可解释的语言属性的表示和使用方面来研究LLM。基于能力的语言模型分析框架旨在通过因果探究干预模型对不同语言属性的内部表征，并测量模型在这些干预下与给定任务的基本事实因果模型的一致性，从而考察特定任务背景下的语言学习能力。我们还开发了一种使用基于梯度的对抗性攻击来执行因果探测干预的新方法，该方法可以针对比现有技术更广泛的属性和表示。最后，我们使用这些干预手段对CAMLE进行了个案研究，分析和比较了不同词汇推理任务的LLM能力，结果表明CAMPE可以用来解释和预测这些任务中的行为。



## **2. Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks**

克服一切困难：克服多语言嵌入倒置攻击中的类型学、脚本和语言混乱 cs.CL

11 pages, 4 figures, 7 tables

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11749v1) [paper-pdf](http://arxiv.org/pdf/2408.11749v1)

**Authors**: Yiyi Chen, Russa Biswas, Heather Lent, Johannes Bjerva

**Abstract**: Large Language Models (LLMs) are susceptible to malicious influence by cyber attackers through intrusions such as adversarial, backdoor, and embedding inversion attacks. In response, the burgeoning field of LLM Security aims to study and defend against such threats. Thus far, the majority of works in this area have focused on monolingual English models, however, emerging research suggests that multilingual LLMs may be more vulnerable to various attacks than their monolingual counterparts. While previous work has investigated embedding inversion over a small subset of European languages, it is challenging to extrapolate these findings to languages from different linguistic families and with differing scripts. To this end, we explore the security of multilingual LLMs in the context of embedding inversion attacks and investigate cross-lingual and cross-script inversion across 20 languages, spanning over 8 language families and 12 scripts. Our findings indicate that languages written in Arabic script and Cyrillic script are particularly vulnerable to embedding inversion, as are languages within the Indo-Aryan language family. We further observe that inversion models tend to suffer from language confusion, sometimes greatly reducing the efficacy of an attack. Accordingly, we systematically explore this bottleneck for inversion models, uncovering predictable patterns which could be leveraged by attackers. Ultimately, this study aims to further the field's understanding of the outstanding security vulnerabilities facing multilingual LLMs and raise awareness for the languages most at risk of negative impact from these attacks.

摘要: 大型语言模型(LLM)容易受到网络攻击者通过对抗性、后门和嵌入反转攻击等入侵的恶意影响。作为回应，LLM Security这个新兴领域的目标是研究和防御此类威胁。到目前为止，这一领域的研究大多集中在单语英语模型上，然而，新的研究表明，多语种的LLM可能比单语的LLM更容易受到各种攻击。虽然以前的工作已经研究了在一小部分欧洲语言上嵌入倒置，但将这些发现外推到来自不同语系和不同脚本的语言是具有挑战性的。为此，我们在嵌入倒置攻击的情况下探索了多语言LLMS的安全性，并研究了跨语言和跨脚本的跨语言和跨脚本倒置，涉及8个语系和12个脚本。我们的发现表明，用阿拉伯文字和西里尔文字书写的语言特别容易嵌入倒置，印度-雅利安语系的语言也是如此。我们进一步观察到，倒置模型往往受到语言混乱的影响，有时会极大地降低攻击的有效性。因此，我们系统地探索了倒置模型的这一瓶颈，揭示了可被攻击者利用的可预测模式。最终，这项研究旨在加深外地对多语种土地管理系统面临的突出安全漏洞的了解，并提高对最有可能受到这些攻击的负面影响的语言的认识。



## **3. First line of defense: A robust first layer mitigates adversarial attacks**

第一道防线：强大的第一层减轻对抗攻击 cs.LG

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11680v1) [paper-pdf](http://arxiv.org/pdf/2408.11680v1)

**Authors**: Janani Suresh, Nancy Nayak, Sheetal Kalyani

**Abstract**: Adversarial training (AT) incurs significant computational overhead, leading to growing interest in designing inherently robust architectures. We demonstrate that a carefully designed first layer of the neural network can serve as an implicit adversarial noise filter (ANF). This filter is created using a combination of large kernel size, increased convolution filters, and a maxpool operation. We show that integrating this filter as the first layer in architectures such as ResNet, VGG, and EfficientNet results in adversarially robust networks. Our approach achieves higher adversarial accuracies than existing natively robust architectures without AT and is competitive with adversarial-trained architectures across a wide range of datasets. Supporting our findings, we show that (a) the decision regions for our method have better margins, (b) the visualized loss surfaces are smoother, (c) the modified peak signal-to-noise ratio (mPSNR) values at the output of the ANF are higher, (d) high-frequency components are more attenuated, and (e) architectures incorporating ANF exhibit better denoising in Gaussian noise compared to baseline architectures. Code for all our experiments are available at \url{https://github.com/janani-suresh-97/first-line-defence.git}.

摘要: 对抗训练(AT)带来了巨大的计算开销，导致人们对设计具有内在健壮性的体系结构的兴趣与日俱增。我们证明了精心设计的第一层神经网络可以用作隐式对抗性噪声过滤器(ANF)。该过滤器使用较大的内核大小、增加的卷积过滤器和最大池操作的组合来创建。我们表明，在ResNet、VGG和EfficientNet等体系结构中集成该过滤器作为第一层会产生相反的健壮性网络。我们的方法获得了比现有的没有AT的本地健壮体系结构更高的对抗准确率，并且在广泛的数据集上与经过对抗训练的体系结构具有竞争力。支持我们的发现，我们的结果表明：(A)我们的方法的判决区域具有更好的裕度，(B)可视化的损失表面更平滑，(C)ANF输出的修正峰值信噪比(MPSNR)值更高，(D)高频分量更弱，(E)与基线结构相比，结合ANF的结构在高斯噪声中表现出更好的去噪效果。我们所有实验的代码都可以在\url{https://github.com/janani-suresh-97/first-line-defence.git}.上找到



## **4. Latent Feature and Attention Dual Erasure Attack against Multi-View Diffusion Models for 3D Assets Protection**

针对3D资产保护的多视图扩散模型的潜在特征和注意力双重擦除攻击 cs.CV

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11408v1) [paper-pdf](http://arxiv.org/pdf/2408.11408v1)

**Authors**: Jingwei Sun, Xuchong Zhang, Changfeng Sun, Qicheng Bai, Hongbin Sun

**Abstract**: Multi-View Diffusion Models (MVDMs) enable remarkable improvements in the field of 3D geometric reconstruction, but the issue regarding intellectual property has received increasing attention due to unauthorized imitation. Recently, some works have utilized adversarial attacks to protect copyright. However, all these works focus on single-image generation tasks which only need to consider the inner feature of images. Previous methods are inefficient in attacking MVDMs because they lack the consideration of disrupting the geometric and visual consistency among the generated multi-view images. This paper is the first to address the intellectual property infringement issue arising from MVDMs. Accordingly, we propose a novel latent feature and attention dual erasure attack to disrupt the distribution of latent feature and the consistency across the generated images from multi-view and multi-domain simultaneously. The experiments conducted on SOTA MVDMs indicate that our approach achieves superior performances in terms of attack effectiveness, transferability, and robustness against defense methods. Therefore, this paper provides an efficient solution to protect 3D assets from MVDMs-based 3D geometry reconstruction.

摘要: 多视点扩散模型(MVDM)在三维几何重建领域取得了显著的进步，但由于未经授权的仿制，涉及知识产权的问题也越来越受到关注。最近，一些作品利用对抗性攻击来保护版权。然而，这些工作都集中在单幅图像生成任务上，只需要考虑图像的内部特征。以前的方法在攻击MVDM时效率不高，因为它们没有考虑破坏生成的多视角图像之间的几何和视觉一致性。这是第一篇关于MVDM引起的知识产权侵权问题的论文。因此，我们提出了一种新的潜在特征和注意双重擦除攻击，以同时扰乱潜在特征的分布和多视角、多领域生成图像的一致性。在Sota MVDM上进行的实验表明，我们的方法在攻击有效性、可转移性和对防御方法的健壮性方面取得了优越的性能。因此，本文为保护3D资产免受基于MVDM的3D几何重建提供了一种有效的解决方案。



## **5. AntifakePrompt: Prompt-Tuned Vision-Language Models are Fake Image Detectors**

AntifakePrompt：预算调整的视觉语言模型是假图像检测器 cs.CV

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2310.17419v3) [paper-pdf](http://arxiv.org/pdf/2310.17419v3)

**Authors**: You-Ming Chang, Chen Yeh, Wei-Chen Chiu, Ning Yu

**Abstract**: Deep generative models can create remarkably photorealistic fake images while raising concerns about misinformation and copyright infringement, known as deepfake threats. Deepfake detection technique is developed to distinguish between real and fake images, where the existing methods typically learn classifiers in the image domain or various feature domains. However, the generalizability of deepfake detection against emerging and more advanced generative models remains challenging. In this paper, being inspired by the zero-shot advantages of Vision-Language Models (VLMs), we propose a novel approach called AntifakePrompt, using VLMs (e.g., InstructBLIP) and prompt tuning techniques to improve the deepfake detection accuracy over unseen data. We formulate deepfake detection as a visual question answering problem, and tune soft prompts for InstructBLIP to answer the real/fake information of a query image. We conduct full-spectrum experiments on datasets from a diversity of 3 held-in and 20 held-out generative models, covering modern text-to-image generation, image editing and adversarial image attacks. These testing datasets provide useful benchmarks in the realm of deepfake detection for further research. Moreover, results demonstrate that (1) the deepfake detection accuracy can be significantly and consistently improved (from 71.06% to 92.11%, in average accuracy over unseen domains) using pretrained vision-language models with prompt tuning; (2) our superior performance is at less cost of training data and trainable parameters, resulting in an effective and efficient solution for deepfake detection. Code and models can be found at https://github.com/nctu-eva-lab/AntifakePrompt.

摘要: 深度生成模型可以创建非常逼真的虚假图像，同时引发人们对错误信息和侵犯版权的担忧，即所谓的深度虚假威胁。深伪检测技术是为了区分真实和虚假的图像而发展起来的，现有的方法通常在图像域或各种特征域学习分类器。然而，针对新兴的和更高级的生成模型的深伪检测的泛化能力仍然具有挑战性。受视觉语言模型(VLMS)零射优势的启发，本文提出了一种新的基于视觉语言模型(VLMS，InstructBLIP)和提示调优的方法，以提高对不可见数据的深度伪检测精度。我们将深度伪检测描述为一个视觉问答问题，并对InstructBLIP的软提示进行调整，以回答查询图像的真假信息。我们在来自3个坚持和20个坚持的生成模型的数据集上进行了全谱实验，涵盖了现代文本到图像的生成、图像编辑和对抗性图像攻击。这些测试数据集为深度伪检测领域的进一步研究提供了有用的基准。实验结果表明：(1)通过快速调整预先训练的视觉语言模型，深度伪检测的正确率可以从71.06%提高到92.11%；(2)我们的优越性能是以较少的训练数据和可训练的参数为代价的，从而为深度伪检测提供了一个有效和高效的解决方案。代码和模型可在https://github.com/nctu-eva-lab/AntifakePrompt.上找到



## **6. Steering cooperation: Adversarial attacks on prisoner's dilemma in complex networks**

指导合作：对复杂网络中囚犯困境的对抗攻击 physics.soc-ph

17 pages, 4 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2406.19692v3) [paper-pdf](http://arxiv.org/pdf/2406.19692v3)

**Authors**: Kazuhiro Takemoto

**Abstract**: This study examines the application of adversarial attack concepts to control the evolution of cooperation in the prisoner's dilemma game in complex networks. Specifically, it proposes a simple adversarial attack method that drives players' strategies towards a target state by adding small perturbations to social networks. The proposed method is evaluated on both model and real-world networks. Numerical simulations demonstrate that the proposed method can effectively promote cooperation with significantly smaller perturbations compared to other techniques. Additionally, this study shows that adversarial attacks can also be useful in inhibiting cooperation (promoting defection). The findings reveal that adversarial attacks on social networks can be potent tools for both promoting and inhibiting cooperation, opening new possibilities for controlling cooperative behavior in social systems while also highlighting potential risks.

摘要: 本研究探讨了对抗攻击概念在复杂网络中囚犯困境游戏中控制合作演变的应用。具体来说，它提出了一种简单的对抗攻击方法，通过向社交网络添加小扰动来推动玩家的策略走向目标状态。在模型和现实世界网络上对所提出的方法进行了评估。数值模拟表明，与其他技术相比，所提出的方法可以有效地促进协作，且扰动要小得多。此外，这项研究表明，对抗性攻击也可能有助于抑制合作（促进叛逃）。研究结果表明，对社交网络的对抗性攻击可以成为促进和抑制合作的有力工具，为控制社会系统中的合作行为开辟了新的可能性，同时也凸显了潜在的风险。



## **7. Investigating Imperceptibility of Adversarial Attacks on Tabular Data: An Empirical Analysis**

调查表格数据对抗性攻击的不可感知性：实证分析 cs.LG

33 pages

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2407.11463v2) [paper-pdf](http://arxiv.org/pdf/2407.11463v2)

**Authors**: Zhipeng He, Chun Ouyang, Laith Alzubaidi, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks are a potential threat to machine learning models by causing incorrect predictions through imperceptible perturbations to the input data. While these attacks have been extensively studied in unstructured data like images, applying them to tabular data, poses new challenges. These challenges arise from the inherent heterogeneity and complex feature interdependencies in tabular data, which differ from the image data. To account for this distinction, it is necessary to establish tailored imperceptibility criteria specific to tabular data. However, there is currently a lack of standardised metrics for assessing the imperceptibility of adversarial attacks on tabular data. To address this gap, we propose a set of key properties and corresponding metrics designed to comprehensively characterise imperceptible adversarial attacks on tabular data. These are: proximity to the original input, sparsity of altered features, deviation from the original data distribution, sensitivity in perturbing features with narrow distribution, immutability of certain features that should remain unchanged, feasibility of specific feature values that should not go beyond valid practical ranges, and feature interdependencies capturing complex relationships between data attributes. We evaluate the imperceptibility of five adversarial attacks, including both bounded attacks and unbounded attacks, on tabular data using the proposed imperceptibility metrics. The results reveal a trade-off between the imperceptibility and effectiveness of these attacks. The study also identifies limitations in current attack algorithms, offering insights that can guide future research in the area. The findings gained from this empirical analysis provide valuable direction for enhancing the design of adversarial attack algorithms, thereby advancing adversarial machine learning on tabular data.

摘要: 对抗性攻击通过对输入数据的不可察觉的扰动而导致错误的预测，从而对机器学习模型构成潜在的威胁。虽然这些攻击已经在图像等非结构化数据中得到了广泛研究，但将它们应用于表格数据带来了新的挑战。这些挑战源于表格数据固有的异构性和复杂的特征相互依赖关系，而表格数据不同于图像数据。为了说明这一区别，有必要建立专门针对表格数据的不可察觉标准。然而，目前缺乏用于评估对抗性攻击对表格数据的不可感知性的标准化指标。为了弥补这一差距，我们提出了一组关键属性和相应的度量，旨在全面表征对表格数据的不可察觉的对抗性攻击。它们是：接近原始输入、改变特征的稀疏性、偏离原始数据分布、对具有窄分布的扰动特征的敏感性、某些应保持不变的特征的不变性、不应超出有效实际范围的特定特征值的可行性、以及捕捉数据属性之间的复杂关系的特征相互依赖关系。我们使用所提出的不可感知性度量评估了五种对抗性攻击，包括有界攻击和无界攻击对表格数据的不可感知性。结果揭示了这些攻击的隐蔽性和有效性之间的权衡。该研究还确定了当前攻击算法的局限性，提供了可以指导该领域未来研究的见解。这一实证分析的结果为改进对抗性攻击算法的设计，从而推进对抗性表格数据机器学习提供了有价值的指导。



## **8. Unlocking Adversarial Suffix Optimization Without Affirmative Phrases: Efficient Black-box Jailbreaking via LLM as Optimizer**

在没有肯定短语的情况下解锁敌对后缀优化：通过LLM作为优化器的高效黑匣子越狱 cs.AI

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11313v1) [paper-pdf](http://arxiv.org/pdf/2408.11313v1)

**Authors**: Weipeng Jiang, Zhenting Wang, Juan Zhai, Shiqing Ma, Zhengyu Zhao, Chao Shen

**Abstract**: Despite prior safety alignment efforts, mainstream LLMs can still generate harmful and unethical content when subjected to jailbreaking attacks. Existing jailbreaking methods fall into two main categories: template-based and optimization-based methods. The former requires significant manual effort and domain knowledge, while the latter, exemplified by Greedy Coordinate Gradient (GCG), which seeks to maximize the likelihood of harmful LLM outputs through token-level optimization, also encounters several limitations: requiring white-box access, necessitating pre-constructed affirmative phrase, and suffering from low efficiency. In this paper, we present ECLIPSE, a novel and efficient black-box jailbreaking method utilizing optimizable suffixes. Drawing inspiration from LLMs' powerful generation and optimization capabilities, we employ task prompts to translate jailbreaking goals into natural language instructions. This guides the LLM to generate adversarial suffixes for malicious queries. In particular, a harmfulness scorer provides continuous feedback, enabling LLM self-reflection and iterative optimization to autonomously and efficiently produce effective suffixes. Experimental results demonstrate that ECLIPSE achieves an average attack success rate (ASR) of 0.92 across three open-source LLMs and GPT-3.5-Turbo, significantly surpassing GCG in 2.4 times. Moreover, ECLIPSE is on par with template-based methods in ASR while offering superior attack efficiency, reducing the average attack overhead by 83%.

摘要: 尽管之前做出了安全调整的努力，但主流LLM在受到越狱攻击时仍然会产生有害和不道德的内容。现有的越狱方法主要分为两类：基于模板的方法和基于优化的方法。前者需要大量的人工工作和领域知识，而后者，例如贪婪坐标梯度(GCG)，试图通过令牌级优化最大化有害的LLM输出的可能性，也遇到了几个限制：需要白盒访问，必须预先构建肯定短语，以及效率低下。在本文中，我们提出了一种利用可优化后缀的新颖高效的黑盒越狱方法--ECLIPSE。从LLMS强大的生成和优化能力中获得灵感，我们使用任务提示将越狱目标转换为自然语言指令。这将引导LLM为恶意查询生成敌意后缀。特别是，危害评分器提供持续的反馈，使LLM自我反省和迭代优化能够自主和高效地产生有效的后缀。实验结果表明，ECLIPSE在三个开源LLMS和GPT-3.5-Turbo上的平均攻击成功率(ASR)为0.92，显著超过GCG的2.4倍。此外，在ASR中，eclipse与基于模板的方法不相上下，同时提供了优越的攻击效率，将平均攻击开销降低了83%。



## **9. EEG-Defender: Defending against Jailbreak through Early Exit Generation of Large Language Models**

EEG-Defender：通过早期退出生成大型语言模型来抵御越狱 cs.AI

19 pages, 7 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11308v1) [paper-pdf](http://arxiv.org/pdf/2408.11308v1)

**Authors**: Chongwen Zhao, Zhihao Dou, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. Built upon this idea, we introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85\% in comparison with 50\% for the present SOTAs, with minimal impact on the utility and effectiveness of LLMs.

摘要: 大语言模型在各种应用中日益引起人们的关注。尽管如此，随着一些用户试图利用这些模型达到恶意目的，包括合成受控物质和传播虚假信息，人们越来越担心。为了减轻这种风险，人们提出了“对准”技术的概念。然而，最近的研究表明，这种对齐可以使用复杂的即时工程或敌对后缀来破坏，这是一种被称为“越狱”的技术。我们的研究从LLMS类似人类的生成过程中获得了线索。我们发现，虽然越狱提示可能会产生类似于良性提示的输出日志，但它们在模型潜在空间中的初始嵌入往往更类似于恶意提示。利用这一发现，我们建议利用LLMS的早期变压器输出作为一种手段来检测恶意输入，并立即终止生成。基于这一想法，我们介绍了一种简单但重要的防御方法，称为用于LLMS的EEG-Defender。我们在三个模型上对十种越狱方法进行了全面的实验。我们的结果表明，EEG-Defender能够显著降低攻击成功率(ASR)，与现有SOTAS的50%相比，约为85%，而对LLMS的实用性和有效性的影响最小。



## **10. Correlation Analysis of Adversarial Attack in Time Series Classification**

时间序列分类中对抗性攻击的相关性分析 cs.LG

15 pages, 7 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11264v1) [paper-pdf](http://arxiv.org/pdf/2408.11264v1)

**Authors**: Zhengyang Li, Wenhao Liang, Chang Dong, Weitong Chen, Dong Huang

**Abstract**: This study investigates the vulnerability of time series classification models to adversarial attacks, with a focus on how these models process local versus global information under such conditions. By leveraging the Normalized Auto Correlation Function (NACF), an exploration into the inclination of neural networks is conducted. It is demonstrated that regularization techniques, particularly those employing Fast Fourier Transform (FFT) methods and targeting frequency components of perturbations, markedly enhance the effectiveness of attacks. Meanwhile, the defense strategies, like noise introduction and Gaussian filtering, are shown to significantly lower the Attack Success Rate (ASR), with approaches based on noise introducing notably effective in countering high-frequency distortions. Furthermore, models designed to prioritize global information are revealed to possess greater resistance to adversarial manipulations. These results underline the importance of designing attack and defense mechanisms, informed by frequency domain analysis, as a means to considerably reinforce the resilience of neural network models against adversarial threats.

摘要: 本文研究了时间序列分类模型对敌意攻击的脆弱性，重点研究了在这种情况下这些模型是如何处理局部和全局信息的。利用归一化自相关函数(NACF)对神经网络的倾向性进行了探讨。结果表明，正则化技术，特别是利用快速傅立叶变换(FFT)方法和针对扰动的频率分量的正则化技术，显著地提高了攻击的有效性。同时，噪声引入和高斯滤波等防御策略显著降低了攻击成功率，其中基于噪声引入的防御策略在对抗高频失真方面效果显著。此外，旨在对全球信息进行优先排序的模型被揭示出对对手操纵具有更强的抵抗力。这些结果强调了通过频域分析设计攻击和防御机制的重要性，以此作为显著增强神经网络模型对对手威胁的弹性的一种手段。



## **11. Revisiting Min-Max Optimization Problem in Adversarial Training**

重温对抗训练中的最小-最大优化问题 cs.CV

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.11218v1) [paper-pdf](http://arxiv.org/pdf/2408.11218v1)

**Authors**: Sina Hajer Ahmadi, Hassan Bahrami

**Abstract**: The rise of computer vision applications in the real world puts the security of the deep neural networks at risk. Recent works demonstrate that convolutional neural networks are susceptible to adversarial examples - where the input images look similar to the natural images but are classified incorrectly by the model. To provide a rebuttal to this problem, we propose a new method to build robust deep neural networks against adversarial attacks by reformulating the saddle point optimization problem in \cite{madry2017towards}. Our proposed method offers significant resistance and a concrete security guarantee against multiple adversaries. The goal of this paper is to act as a stepping stone for a new variation of deep learning models which would lead towards fully robust deep learning models.

摘要: 现实世界中计算机视觉应用的兴起使深度神经网络的安全面临风险。最近的工作表明，卷积神经网络容易受到对抗性示例的影响--其中输入图像看起来与自然图像相似，但模型分类错误。为了反驳这个问题，我们提出了一种新的方法，通过重新定义\cite{madry 2017 toward}中的鞍点优化问题来构建稳健的深度神经网络来对抗对抗攻击。我们提出的方法提供了针对多个对手的显着的抵抗力和具体的安全保证。本文的目标是成为深度学习模型新变体的垫脚石，这将导致完全稳健的深度学习模型。



## **12. GAIM: Attacking Graph Neural Networks via Adversarial Influence Maximization**

GAIM：通过对抗影响最大化攻击图神经网络 cs.LG

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10948v1) [paper-pdf](http://arxiv.org/pdf/2408.10948v1)

**Authors**: Xiaodong Yang, Xiaoting Li, Huiyuan Chen, Yiwei Cai

**Abstract**: Recent studies show that well-devised perturbations on graph structures or node features can mislead trained Graph Neural Network (GNN) models. However, these methods often overlook practical assumptions, over-rely on heuristics, or separate vital attack components. In response, we present GAIM, an integrated adversarial attack method conducted on a node feature basis while considering the strict black-box setting. Specifically, we define an adversarial influence function to theoretically assess the adversarial impact of node perturbations, thereby reframing the GNN attack problem into the adversarial influence maximization problem. In our approach, we unify the selection of the target node and the construction of feature perturbations into a single optimization problem, ensuring a unique and consistent feature perturbation for each target node. We leverage a surrogate model to transform this problem into a solvable linear programming task, streamlining the optimization process. Moreover, we extend our method to accommodate label-oriented attacks, broadening its applicability. Thorough evaluations on five benchmark datasets across three popular models underscore the effectiveness of our method in both untargeted and label-oriented targeted attacks. Through comprehensive analysis and ablation studies, we demonstrate the practical value and efficacy inherent to our design choices.

摘要: 最近的研究表明，对图结构或节点特征的精心设计的扰动会误导训练好的图神经网络(GNN)模型。然而，这些方法往往忽略了实际的假设，过度依赖启发式方法，或者分离出重要的攻击组件。对此，我们提出了一种基于节点特征的综合对抗性攻击方法GAIM，同时考虑了严格的黑盒设置。具体地说，我们定义了一个对抗性影响函数来从理论上评估节点扰动的对抗性影响，从而将GNN攻击问题重组为对抗性影响最大化问题。在我们的方法中，我们将目标节点的选择和特征扰动的构造统一为一个优化问题，确保每个目标节点具有唯一和一致的特征扰动。我们利用代理模型将这个问题转化为一个可解的线性规划任务，从而简化了优化过程。此外，我们还扩展了我们的方法以适应面向标签的攻击，从而扩大了它的适用性。对三个流行模型上的五个基准数据集进行的全面评估强调了我们的方法在非目标攻击和面向标签的目标攻击中的有效性。通过综合分析和烧蚀研究，我们证明了我们的设计选择所固有的实用价值和功效。



## **13. A Grey-box Attack against Latent Diffusion Model-based Image Editing by Posterior Collapse**

对基于潜在扩散模型的图像编辑的灰箱攻击 cs.CV

21 pages, 7 figures, 10 tables

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10901v1) [paper-pdf](http://arxiv.org/pdf/2408.10901v1)

**Authors**: Zhongliang Guo, Lei Fang, Jingyu Lin, Yifei Qian, Shuai Zhao, Zeyu Wang, Junhao Dong, Cunjian Chen, Ognjen Arandjelović, Chun Pong Lau

**Abstract**: Recent advancements in generative AI, particularly Latent Diffusion Models (LDMs), have revolutionized image synthesis and manipulation. However, these generative techniques raises concerns about data misappropriation and intellectual property infringement. Adversarial attacks on machine learning models have been extensively studied, and a well-established body of research has extended these techniques as a benign metric to prevent the underlying misuse of generative AI. Current approaches to safeguarding images from manipulation by LDMs are limited by their reliance on model-specific knowledge and their inability to significantly degrade semantic quality of generated images. In response to these shortcomings, we propose the Posterior Collapse Attack (PCA) based on the observation that VAEs suffer from posterior collapse during training. Our method minimizes dependence on the white-box information of target models to get rid of the implicit reliance on model-specific knowledge. By accessing merely a small amount of LDM parameters, in specific merely the VAE encoder of LDMs, our method causes a substantial semantic collapse in generation quality, particularly in perceptual consistency, and demonstrates strong transferability across various model architectures. Experimental results show that PCA achieves superior perturbation effects on image generation of LDMs with lower runtime and VRAM. Our method outperforms existing techniques, offering a more robust and generalizable solution that is helpful in alleviating the socio-technical challenges posed by the rapidly evolving landscape of generative AI.

摘要: 生成性人工智能的最新进展，特别是潜在扩散模型(LDM)，已经彻底改变了图像合成和处理。然而，这些生成性技术引发了人们对数据挪用和侵犯知识产权的担忧。对机器学习模型的对抗性攻击已经被广泛研究，一系列成熟的研究已经将这些技术扩展为一种良性的衡量标准，以防止潜在的生成性人工智能的滥用。当前保护图像免受LDM操纵的方法受到它们对模型特定知识的依赖以及它们无法显著降低所生成图像的语义质量的限制。针对这些不足，我们提出了后部塌陷攻击(PCA)，基于VAE在训练过程中遭受后部塌陷的观察。我们的方法最大限度地减少了对目标模型白盒信息的依赖，摆脱了对特定模型知识的隐含依赖。通过只访问少量的LDM参数，特别是LDM的VAE编码器，我们的方法导致生成质量的语义崩溃，特别是在感知一致性方面，并表现出强大的跨模型体系结构的可移植性。实验结果表明，主成分分析算法以较低的运行时间和较低的VRAM实现了较好的图像生成扰动效果。我们的方法优于现有的技术，提供了一个更健壮和更具通用性的解决方案，有助于缓解快速发展的生成性人工智能所带来的社会技术挑战。



## **14. Towards Efficient Formal Verification of Spiking Neural Network**

尖峰神经网络的有效形式验证 cs.AI

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10900v1) [paper-pdf](http://arxiv.org/pdf/2408.10900v1)

**Authors**: Baekryun Seong, Jieung Kim, Sang-Ki Ko

**Abstract**: Recently, AI research has primarily focused on large language models (LLMs), and increasing accuracy often involves scaling up and consuming more power. The power consumption of AI has become a significant societal issue; in this context, spiking neural networks (SNNs) offer a promising solution. SNNs operate event-driven, like the human brain, and compress information temporally. These characteristics allow SNNs to significantly reduce power consumption compared to perceptron-based artificial neural networks (ANNs), highlighting them as a next-generation neural network technology. However, societal concerns regarding AI go beyond power consumption, with the reliability of AI models being a global issue. For instance, adversarial attacks on AI models are a well-studied problem in the context of traditional neural networks. Despite their importance, the stability and property verification of SNNs remains in the early stages of research. Most SNN verification methods are time-consuming and barely scalable, making practical applications challenging. In this paper, we introduce temporal encoding to achieve practical performance in verifying the adversarial robustness of SNNs. We conduct a theoretical analysis of this approach and demonstrate its success in verifying SNNs at previously unmanageable scales. Our contribution advances SNN verification to a practical level, facilitating the safer application of SNNs.

摘要: 最近，人工智能的研究主要集中在大型语言模型(LLM)上，而提高精确度往往需要扩大规模和消耗更多功率。人工智能的能耗已经成为一个重要的社会问题；在这种背景下，尖峰神经网络(SNN)提供了一个有前途的解决方案。SNN像人脑一样，以事件驱动的方式运行，并在时间上压缩信息。与基于感知器的人工神经网络(ANN)相比，这些特性使SNN能够显著降低功耗，突出了它们作为下一代神经网络技术的重要性。然而，社会对人工智能的担忧不仅仅是电力消耗，人工智能模型的可靠性是一个全球问题。例如，在传统神经网络的背景下，对人工智能模型的敌意攻击是一个研究得很好的问题。尽管它们很重要，但SNN的稳定性和性质验证仍处于研究的早期阶段。大多数SNN验证方法都很耗时且几乎不可扩展，这给实际应用带来了挑战。在本文中，我们引入时间编码以达到在验证SNN的对抗健壮性方面的实际性能。我们对这种方法进行了理论分析，并证明了它在以前无法管理的规模上验证SNN的成功。我们的贡献将SNN验证提升到了一个实用的水平，促进了SNN的更安全应用。



## **15. Exploiting Defenses against GAN-Based Feature Inference Attacks in Federated Learning**

在联邦学习中利用防御基于GAN的特征推理攻击 cs.CR

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2004.12571v3) [paper-pdf](http://arxiv.org/pdf/2004.12571v3)

**Authors**: Xinjian Luo, Xianglong Zhang

**Abstract**: Federated learning (FL) is a decentralized model training framework that aims to merge isolated data islands while maintaining data privacy. However, recent studies have revealed that Generative Adversarial Network (GAN) based attacks can be employed in FL to learn the distribution of private datasets and reconstruct recognizable images. In this paper, we exploit defenses against GAN-based attacks in FL and propose a framework, Anti-GAN, to prevent attackers from learning the real distribution of the victim's data. The core idea of Anti-GAN is to manipulate the visual features of private training images to make them indistinguishable to human eyes even restored by attackers. Specifically, Anti-GAN projects the private dataset onto a GAN's generator and combines the generated fake images with the actual images to create the training dataset, which is then used for federated model training. The experimental results demonstrate that Anti-GAN is effective in preventing attackers from learning the distribution of private images while causing minimal harm to the accuracy of the federated model.

摘要: 联邦学习(FL)是一种去中心化的模型训练框架，旨在合并孤立的数据孤岛，同时保持数据隐私。然而，最近的研究表明，基于生成性对抗网络(GAN)的攻击可以用于FL中，以学习私有数据集的分布并重建可识别的图像。在本文中，我们在FL中利用对基于GAN的攻击的防御，并提出了一个框架--Anti-GAN，以防止攻击者了解受害者数据的真实分布。Anti-GAN的核心思想是操纵私人训练图像的视觉特征，使其即使被攻击者恢复也无法辨别人眼。具体地说，Anti-GAN将私有数据集投影到GAN的生成器上，并将生成的虚假图像与实际图像相结合来创建训练数据集，然后将其用于联合模型训练。实验结果表明，该算法能有效地防止攻击者学习私有图像的分布，同时对联邦模型的准确性造成最小的损害。



## **16. Honeyquest: Rapidly Measuring the Enticingness of Cyber Deception Techniques with Code-based Questionnaires**

Honeyquest：使用基于代码的数据库快速衡量网络欺骗技术的吸引力 cs.CR

to be published in the 27th International Symposium on Research in  Attacks, Intrusions and Defenses (RAID 2024), dataset and source code  available at https://github.com/dynatrace-oss/honeyquest

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10796v1) [paper-pdf](http://arxiv.org/pdf/2408.10796v1)

**Authors**: Mario Kahlhofer, Stefan Achleitner, Stefan Rass, René Mayrhofer

**Abstract**: Fooling adversaries with traps such as honeytokens can slow down cyber attacks and create strong indicators of compromise. Unfortunately, cyber deception techniques are often poorly specified. Also, realistically measuring their effectiveness requires a well-exposed software system together with a production-ready implementation of these techniques. This makes rapid prototyping challenging. Our work translates 13 previously researched and 12 self-defined techniques into a high-level, machine-readable specification. Our open-source tool, Honeyquest, allows researchers to quickly evaluate the enticingness of deception techniques without implementing them. We test the enticingness of 25 cyber deception techniques and 19 true security risks in an experiment with 47 humans. We successfully replicate the goals of previous work with many consistent findings, but without a time-consuming implementation of these techniques on real computer systems. We provide valuable insights for the design of enticing deception and also show that the presence of cyber deception can significantly reduce the risk that adversaries will find a true security risk by about 22% on average.

摘要: 用蜜令牌等陷阱愚弄对手可以减缓网络攻击，并创造出强大的妥协迹象。不幸的是，网络欺骗技术往往没有得到很好的说明。此外，现实地衡量它们的有效性需要一个暴露良好的软件系统，以及这些技术的生产就绪实现。这使得快速原型制作具有挑战性。我们的工作将之前研究的13种技术和12种自定义技术转换为机器可读的高级规范。我们的开源工具HoneyQuest允许研究人员在不实施欺骗技术的情况下快速评估它们的诱惑力。我们在47个人的实验中测试了25种网络欺骗技术和19种真正的安全风险的诱惑力。我们成功地复制了以前工作的目标，有许多一致的发现，但没有在真实的计算机系统上耗时地实施这些技术。我们为诱人欺骗的设计提供了有价值的见解，并表明网络欺骗的存在可以显著降低攻击者发现真正安全风险的风险，平均约为22%。



## **17. Adversarial Attack for Explanation Robustness of Rationalization Models**

对合理化模型解释稳健性的对抗攻击 cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10795v1) [paper-pdf](http://arxiv.org/pdf/2408.10795v1)

**Authors**: Yuankai Zhang, Lingxiao Kong, Haozhao Wang, Ruixuan Li, Jun Wang, Yuhua Li, Wei Liu

**Abstract**: Rationalization models, which select a subset of input text as rationale-crucial for humans to understand and trust predictions-have recently emerged as a prominent research area in eXplainable Artificial Intelligence. However, most of previous studies mainly focus on improving the quality of the rationale, ignoring its robustness to malicious attack. Specifically, whether the rationalization models can still generate high-quality rationale under the adversarial attack remains unknown. To explore this, this paper proposes UAT2E, which aims to undermine the explainability of rationalization models without altering their predictions, thereby eliciting distrust in these models from human users. UAT2E employs the gradient-based search on triggers and then inserts them into the original input to conduct both the non-target and target attack. Experimental results on five datasets reveal the vulnerability of rationalization models in terms of explanation, where they tend to select more meaningless tokens under attacks. Based on this, we make a series of recommendations for improving rationalization models in terms of explanation.

摘要: 合理化模型选择输入文本的一个子集作为理论基础--这对人类理解和信任预测至关重要--最近已成为可解释人工智能的一个重要研究领域。然而，以往的研究大多侧重于提高理论基础的质量，而忽略了其对恶意攻击的健壮性。具体地说，在对抗性攻击下，合理化模型是否仍能产生高质量的推理仍是未知的。为了探索这一点，本文提出了UAT2E，其目的是在不改变其预测的情况下削弱合理化模型的可解释性，从而引起人类用户对这些模型的不信任。UAT2E在触发器上采用基于梯度的搜索，然后将它们插入到原始输入中，以进行非目标攻击和目标攻击。在五个数据集上的实验结果揭示了合理化模型在解释方面的脆弱性，在攻击下，它们倾向于选择更多无意义的标记。在此基础上，本文从解释的角度提出了一系列改进合理化模型的建议。



## **18. SAM Meets UAP: Attacking Segment Anything Model With Universal Adversarial Perturbation**

Sam会见UAP：用普遍对抗扰动攻击细分任何模型 cs.CV

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2310.12431v2) [paper-pdf](http://arxiv.org/pdf/2310.12431v2)

**Authors**: Dongshen Han, Chaoning Zhang, Sheng Zheng, Chang Lu, Yang Yang, Heng Tao Shen

**Abstract**: As Segment Anything Model (SAM) becomes a popular foundation model in computer vision, its adversarial robustness has become a concern that cannot be ignored. This works investigates whether it is possible to attack SAM with image-agnostic Universal Adversarial Perturbation (UAP). In other words, we seek a single perturbation that can fool the SAM to predict invalid masks for most (if not all) images. We demonstrate convetional image-centric attack framework is effective for image-independent attacks but fails for universal adversarial attack. To this end, we propose a novel perturbation-centric framework that results in a UAP generation method based on self-supervised contrastive learning (CL), where the UAP is set to the anchor sample and the positive sample is augmented from the UAP. The representations of negative samples are obtained from the image encoder in advance and saved in a memory bank. The effectiveness of our proposed CL-based UAP generation method is validated by both quantitative and qualitative results. On top of the ablation study to understand various components in our proposed method, we shed light on the roles of positive and negative samples in making the generated UAP effective for attacking SAM.

摘要: 随着Segment Anything Model(SAM)成为计算机视觉中一种流行的基础模型，其对抗健壮性已成为一个不容忽视的问题。这项工作调查是否有可能用图像不可知的通用对抗扰动(UAP)来攻击SAM。换句话说，我们寻找一个单一的扰动，它可以愚弄SAM来预测大多数(如果不是全部)图像的无效掩码。我们证明了传递式图像中心攻击框架对于图像无关攻击是有效的，但对于通用对抗性攻击是无效的。为此，我们提出了一种新的以扰动为中心的框架，该框架导致了一种基于自监督对比学习(CL)的UAP生成方法，其中UAP被设置为锚定样本，正样本从UAP增加。事先从图像编码器获取负样本的表示，并将其存储在存储体中。定量和定性结果验证了本文提出的基于CL的UAP生成方法的有效性。在消融研究以了解我们提出的方法中的各个组成部分的基础上，我们阐明了正样本和负样本在使生成的UAP有效攻击SAM方面所起的作用。



## **19. Security Assessment of Hierarchical Federated Deep Learning**

分层联邦深度学习的安全评估 cs.LG

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10752v1) [paper-pdf](http://arxiv.org/pdf/2408.10752v1)

**Authors**: D Alqattan, R Sun, H Liang, G Nicosia, V Snasel, R Ranjan, V Ojha

**Abstract**: Hierarchical federated learning (HFL) is a promising distributed deep learning model training paradigm, but it has crucial security concerns arising from adversarial attacks. This research investigates and assesses the security of HFL using a novel methodology by focusing on its resilience against adversarial attacks inference-time and training-time. Through a series of extensive experiments across diverse datasets and attack scenarios, we uncover that HFL demonstrates robustness against untargeted training-time attacks due to its hierarchical structure. However, targeted attacks, particularly backdoor attacks, exploit this architecture, especially when malicious clients are positioned in the overlapping coverage areas of edge servers. Consequently, HFL shows a dual nature in its resilience, showcasing its capability to recover from attacks thanks to its hierarchical aggregation that strengthens its suitability for adversarial training, thereby reinforcing its resistance against inference-time attacks. These insights underscore the necessity for balanced security strategies in HFL systems, leveraging their inherent strengths while effectively mitigating vulnerabilities.

摘要: 分层联邦学习(HFL)是一种很有前途的分布式深度学习模型训练范型，但它存在来自对手攻击的严重安全问题。本研究采用一种新的方法对HFL的安全性进行了研究和评估，重点考察了HFL对对手攻击的恢复能力、推理时间和训练时间。通过一系列针对不同数据集和攻击场景的广泛实验，我们发现HFL由于其层次结构而表现出对非目标训练时间攻击的健壮性。然而，有针对性的攻击，特别是后门攻击，会利用这种体系结构，特别是当恶意客户端位于边缘服务器的重叠覆盖区域时。因此，HFL在其韧性方面表现出双重性质，由于其分层聚集增强了其对对手训练的适宜性，从而显示了其从攻击中恢复的能力，从而增强了其对推理时间攻击的抵抗力。这些见解强调了在HFL系统中平衡安全战略的必要性，利用它们的固有优势，同时有效地减轻漏洞。



## **20. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

PhishAgent：一种用于网络钓鱼网页检测的鲁棒多模式代理 cs.CR

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10738v1) [paper-pdf](http://arxiv.org/pdf/2408.10738v1)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also encounter notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the top k relevant items from offline knowledge bases, utilizing all available information from a webpage, including logos, HTML, and URLs. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.

摘要: 网络钓鱼攻击是在线安全的主要威胁，利用用户漏洞窃取敏感信息。已经开发了各种方法来对抗网络钓鱼，每一种方法的精确度都不同，但它们也遇到了显著的局限性。在本研究中，我们介绍了PhishAgent，一个结合了广泛工具的多通道代理，将线上和线下知识库与多通道大语言模型(MLLMS)相结合。这一组合导致了更广泛的品牌覆盖，从而提高了品牌认知度和召回率。此外，我们提出了一个多通道信息检索框架，旨在利用网页中的所有可用信息，包括徽标、HTML和URL，从离线知识库中提取前k个相关条目。基于三个真实数据集的实验结果表明，该框架在保持模型效率的同时，显著提高了检测准确率，减少了误报和漏报。此外，PhishAgent对各种类型的对抗性攻击表现出很强的韧性。



## **21. Ferret: Faster and Effective Automated Red Teaming with Reward-Based Scoring Technique**

Ferret：更快、更有效的自动化红色团队，采用基于奖励的评分技术 cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10701v1) [paper-pdf](http://arxiv.org/pdf/2408.10701v1)

**Authors**: Tej Deep Pala, Vernon Y. H. Toh, Rishabh Bhardwaj, Soujanya Poria

**Abstract**: In today's era, where large language models (LLMs) are integrated into numerous real-world applications, ensuring their safety and robustness is crucial for responsible AI usage. Automated red-teaming methods play a key role in this process by generating adversarial attacks to identify and mitigate potential vulnerabilities in these models. However, existing methods often struggle with slow performance, limited categorical diversity, and high resource demands. While Rainbow Teaming, a recent approach, addresses the diversity challenge by framing adversarial prompt generation as a quality-diversity search, it remains slow and requires a large fine-tuned mutator for optimal performance. To overcome these limitations, we propose Ferret, a novel approach that builds upon Rainbow Teaming by generating multiple adversarial prompt mutations per iteration and using a scoring function to rank and select the most effective adversarial prompt. We explore various scoring functions, including reward models, Llama Guard, and LLM-as-a-judge, to rank adversarial mutations based on their potential harm to improve the efficiency of the search for harmful mutations. Our results demonstrate that Ferret, utilizing a reward model as a scoring function, improves the overall attack success rate (ASR) to 95%, which is 46% higher than Rainbow Teaming. Additionally, Ferret reduces the time needed to achieve a 90% ASR by 15.2% compared to the baseline and generates adversarial prompts that are transferable i.e. effective on other LLMs of larger size. Our codes are available at https://github.com/declare-lab/ferret.

摘要: 在当今时代，大型语言模型(LLM)被集成到许多现实世界的应用程序中，确保它们的安全性和健壮性对于负责任的人工智能使用至关重要。自动红团队方法通过生成对抗性攻击来识别和缓解这些模型中的潜在漏洞，从而在这一过程中发挥关键作用。然而，现有的方法往往在性能缓慢、分类多样性有限和资源需求高的情况下苦苦挣扎。虽然彩虹组合是最近的一种方法，通过将敌意提示生成框定为一种质量多样性搜索来解决多样性挑战，但它仍然很慢，需要一个大型微调赋值器来实现最佳性能。为了克服这些局限性，我们提出了一种新的方法--FERRET，它建立在彩虹分组的基础上，通过每次迭代产生多个对抗性提示突变，并使用评分函数来对最有效的对抗性提示进行排序和选择。我们探索了各种评分函数，包括奖励模型、骆驼警卫和LLM作为法官，根据潜在的危害对对手突变进行排名，以提高有害突变的搜索效率。我们的结果表明，利用奖励模型作为得分函数的雪貂，将总体攻击成功率(ASR)提高到95%，比彩虹组合高出46%。此外，与基线相比，雪貂将达到90%的ASR所需的时间减少了15.2%，并生成可转移的对抗性提示，即对其他较大规模的LLM有效。我们的代码可在https://github.com/declare-lab/ferret.上获得



## **22. MsMemoryGAN: A Multi-scale Memory GAN for Palm-vein Adversarial Purification**

MsMemoryGAN：一种用于掌静脉对抗净化的多尺度记忆GAN cs.CV

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10694v1) [paper-pdf](http://arxiv.org/pdf/2408.10694v1)

**Authors**: Huafeng Qin, Yuming Fu, Huiyan Zhang, Mounim A. El-Yacoubi, Xinbo Gao, Qun Song, Jun Wang

**Abstract**: Deep neural networks have recently achieved promising performance in the vein recognition task and have shown an increasing application trend, however, they are prone to adversarial perturbation attacks by adding imperceptible perturbations to the input, resulting in making incorrect recognition. To address this issue, we propose a novel defense model named MsMemoryGAN, which aims to filter the perturbations from adversarial samples before recognition. First, we design a multi-scale autoencoder to achieve high-quality reconstruction and two memory modules to learn the detailed patterns of normal samples at different scales. Second, we investigate a learnable metric in the memory module to retrieve the most relevant memory items to reconstruct the input image. Finally, the perceptional loss is combined with the pixel loss to further enhance the quality of the reconstructed image. During the training phase, the MsMemoryGAN learns to reconstruct the input by merely using fewer prototypical elements of the normal patterns recorded in the memory. At the testing stage, given an adversarial sample, the MsMemoryGAN retrieves its most relevant normal patterns in memory for the reconstruction. Perturbations in the adversarial sample are usually not reconstructed well, resulting in purifying the input from adversarial perturbations. We have conducted extensive experiments on two public vein datasets under different adversarial attack methods to evaluate the performance of the proposed approach. The experimental results show that our approach removes a wide variety of adversarial perturbations, allowing vein classifiers to achieve the highest recognition accuracy.

摘要: 深度神经网络近年来在静脉识别任务中取得了良好的性能，并显示出越来越多的应用趋势，但它们容易受到对抗性的扰动攻击，给输入增加不可察觉的扰动，从而导致错误识别。针对这一问题，我们提出了一种新的防御模型MsMemoyGAN，其目的是在识别之前对对手样本中的扰动进行过滤。首先，我们设计了一个多尺度自动编码器来实现高质量的重建，并设计了两个存储模块来学习正常样本在不同尺度上的详细模式。其次，我们研究了记忆模块中的一个可学习度量，以检索最相关的记忆项来重建输入图像。最后，将感知损失与像素损失相结合，进一步提高重建图像的质量。在训练阶段，MsMemory GAN仅通过使用记忆中记录的较少的正常模式的典型元素来学习重建输入。在测试阶段，给定对抗性样本，MsMemory GAN在存储器中检索其最相关的正常模式以用于重建。对抗性样本中的扰动通常不能很好地重构，导致从对抗性扰动中提纯输入。我们在两个公共静脉数据集上进行了大量的实验，在不同的对抗性攻击方法下评估了该方法的性能。实验结果表明，我们的方法消除了各种各样的对抗性扰动，使静脉分类器获得了最高的识别精度。



## **23. Towards Robust Knowledge Unlearning: An Adversarial Framework for Assessing and Improving Unlearning Robustness in Large Language Models**

迈向稳健的知识去学习：评估和改进大型语言模型中去学习稳健性的对抗框架 cs.CL

13 pages

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10682v1) [paper-pdf](http://arxiv.org/pdf/2408.10682v1)

**Authors**: Hongbang Yuan, Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao

**Abstract**: LLM have achieved success in many fields but still troubled by problematic content in the training corpora. LLM unlearning aims at reducing their influence and avoid undesirable behaviours. However, existing unlearning methods remain vulnerable to adversarial queries and the unlearned knowledge resurfaces after the manually designed attack queries. As part of a red-team effort to proactively assess the vulnerabilities of unlearned models, we design Dynamic Unlearning Attack (DUA), a dynamic and automated framework to attack these models and evaluate their robustness. It optimizes adversarial suffixes to reintroduce the unlearned knowledge in various scenarios. We find that unlearned knowledge can be recovered in $55.2\%$ of the questions, even without revealing the unlearned model's parameters. In response to this vulnerability, we propose Latent Adversarial Unlearning (LAU), a universal framework that effectively enhances the robustness of the unlearned process. It formulates the unlearning process as a min-max optimization problem and resolves it through two stages: an attack stage, where perturbation vectors are trained and added to the latent space of LLMs to recover the unlearned knowledge, and a defense stage, where previously trained perturbation vectors are used to enhance unlearned model's robustness. With our LAU framework, we obtain two robust unlearning methods, AdvGA and AdvNPO. We conduct extensive experiments across multiple unlearning benchmarks and various models, and demonstrate that they improve the unlearning effectiveness by over $53.5\%$, cause only less than a $11.6\%$ reduction in neighboring knowledge, and have almost no impact on the model's general capabilities.

摘要: LLM在许多领域取得了成功，但仍受到培训语料库中有问题的内容的困扰。LLM遗忘的目的是减少他们的影响，避免不良行为。然而，现有的遗忘方法仍然容易受到敌意查询的攻击，在人工设计的攻击查询之后，未学习的知识重新浮出水面。作为红团队主动评估未学习模型漏洞的努力的一部分，我们设计了动态遗忘攻击(DUA)，这是一个动态和自动化的框架来攻击这些模型并评估它们的健壮性。它优化对抗性后缀，在不同的场景中重新引入未学习的知识。我们发现，即使在不透露未学习模型参数的情况下，也可以在$55.2\$的问题中恢复未学习知识。针对这一弱点，我们提出了潜在对抗性遗忘(LAU)，这是一个通用的框架，有效地增强了未学习过程的稳健性。它将无学习过程描述为一个极小极大优化问题，并分两个阶段进行求解：攻击阶段，训练扰动向量并将其添加到LLMS的潜在空间以恢复未学习知识；防御阶段，利用先前训练的扰动向量来增强未学习模型的稳健性。在LAU框架下，我们得到了两种稳健的遗忘方法：AdvGA和AdvNPO。我们在多个遗忘基准和不同的模型上进行了大量的实验，结果表明，它们使遗忘效率提高了53.5美元以上，相邻知识仅减少了不到11.6美元，而对模型的总体性能几乎没有影响。



## **24. Iterative Window Mean Filter: Thwarting Diffusion-based Adversarial Purification**

迭代窗口均值过滤器：阻止基于扩散的对抗净化 cs.CR

Under review

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10673v1) [paper-pdf](http://arxiv.org/pdf/2408.10673v1)

**Authors**: Hanrui Wang, Ruoxi Sun, Cunjian Chen, Minhui Xue, Lay-Ki Soon, Shuo Wang, Zhe Jin

**Abstract**: Face authentication systems have brought significant convenience and advanced developments, yet they have become unreliable due to their sensitivity to inconspicuous perturbations, such as adversarial attacks. Existing defenses often exhibit weaknesses when facing various attack algorithms and adaptive attacks or compromise accuracy for enhanced security. To address these challenges, we have developed a novel and highly efficient non-deep-learning-based image filter called the Iterative Window Mean Filter (IWMF) and proposed a new framework for adversarial purification, named IWMF-Diff, which integrates IWMF and denoising diffusion models. These methods can function as pre-processing modules to eliminate adversarial perturbations without necessitating further modifications or retraining of the target system. We demonstrate that our proposed methodologies fulfill four critical requirements: preserved accuracy, improved security, generalizability to various threats in different settings, and better resistance to adaptive attacks. This performance surpasses that of the state-of-the-art adversarial purification method, DiffPure.

摘要: 人脸认证系统带来了极大的便利和先进的发展，但由于它们对诸如敌意攻击等不起眼的扰动非常敏感，因此变得不可靠。现有的防御在面对各种攻击算法和自适应攻击时往往表现出弱点，或者为了增强安全性而损害准确性。为了应对这些挑战，我们开发了一种新颖高效的基于非深度学习的图像过滤器，称为迭代窗口均值过滤器(IWMF)，并提出了一种结合IWMF和去噪扩散模型的新的对抗性净化框架IWMF-DIFF。这些方法可以作为前处理模块来消除对抗性干扰，而不需要对目标系统进行进一步的修改或重新培训。我们证明了我们提出的方法满足了四个关键要求：保持准确性，提高安全性，对不同环境下的各种威胁具有通用性，以及更好地抵抗自适应攻击。这一性能超过了最先进的对抗性净化方法DiffPure。



## **25. Accelerating the Surrogate Retraining for Poisoning Attacks against Recommender Systems**

加速针对推荐系统中毒攻击的代理人再培训 cs.IR

Accepted by RecSys 2024

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10666v1) [paper-pdf](http://arxiv.org/pdf/2408.10666v1)

**Authors**: Yunfan Wu, Qi Cao, Shuchang Tao, Kaike Zhang, Fei Sun, Huawei Shen

**Abstract**: Recent studies have demonstrated the vulnerability of recommender systems to data poisoning attacks, where adversaries inject carefully crafted fake user interactions into the training data of recommenders to promote target items. Current attack methods involve iteratively retraining a surrogate recommender on the poisoned data with the latest fake users to optimize the attack. However, this repetitive retraining is highly time-consuming, hindering the efficient assessment and optimization of fake users. To mitigate this computational bottleneck and develop a more effective attack in an affordable time, we analyze the retraining process and find that a change in the representation of one user/item will cause a cascading effect through the user-item interaction graph. Under theoretical guidance, we introduce \emph{Gradient Passing} (GP), a novel technique that explicitly passes gradients between interacted user-item pairs during backpropagation, thereby approximating the cascading effect and accelerating retraining. With just a single update, GP can achieve effects comparable to multiple original training iterations. Under the same number of retraining epochs, GP enables a closer approximation of the surrogate recommender to the victim. This more accurate approximation provides better guidance for optimizing fake users, ultimately leading to enhanced data poisoning attacks. Extensive experiments on real-world datasets demonstrate the efficiency and effectiveness of our proposed GP.

摘要: 最近的研究证明了推荐器系统对数据中毒攻击的脆弱性，即攻击者将精心制作的虚假用户交互注入推荐器的训练数据中，以推广目标项目。目前的攻击方法包括用最新的虚假用户迭代地重新训练代理推荐器来优化攻击。然而，这种重复的再培训非常耗时，阻碍了对假冒用户的高效评估和优化。为了缓解这一计算瓶颈，并在负担得起的时间内开发出更有效的攻击，我们分析了再培训过程，发现用户/项目表示的变化将通过用户-项目交互图引起级联效应。在理论指导下，我们引入了一种新的技术--梯度传递(GP)，它在反向传播过程中显式地传递交互用户-项目对之间的梯度，从而近似级联效应并加速再训练。只需一次更新，GP就可以达到与多个原始训练迭代相当的效果。在相同的再培训次数下，GP使代理推荐者与受害者更接近。这种更准确的近似为优化虚假用户提供了更好的指导，最终导致了增强的数据中毒攻击。在真实数据集上的大量实验证明了我们提出的GP的效率和有效性。



## **26. Privacy-preserving Universal Adversarial Defense for Black-box Models**

黑匣子模型的隐私保护通用对抗防御 cs.LG

12 pages, 9 figures

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10647v1) [paper-pdf](http://arxiv.org/pdf/2408.10647v1)

**Authors**: Qiao Li, Cong Wu, Jing Chen, Zijun Zhang, Kun He, Ruiying Du, Xinxin Wang, Qingchuang Zhao, Yang Liu

**Abstract**: Deep neural networks (DNNs) are increasingly used in critical applications such as identity authentication and autonomous driving, where robustness against adversarial attacks is crucial. These attacks can exploit minor perturbations to cause significant prediction errors, making it essential to enhance the resilience of DNNs. Traditional defense methods often rely on access to detailed model information, which raises privacy concerns, as model owners may be reluctant to share such data. In contrast, existing black-box defense methods fail to offer a universal defense against various types of adversarial attacks. To address these challenges, we introduce DUCD, a universal black-box defense method that does not require access to the target model's parameters or architecture. Our approach involves distilling the target model by querying it with data, creating a white-box surrogate while preserving data privacy. We further enhance this surrogate model using a certified defense based on randomized smoothing and optimized noise selection, enabling robust defense against a broad range of adversarial attacks. Comparative evaluations between the certified defenses of the surrogate and target models demonstrate the effectiveness of our approach. Experiments on multiple image classification datasets show that DUCD not only outperforms existing black-box defenses but also matches the accuracy of white-box defenses, all while enhancing data privacy and reducing the success rate of membership inference attacks.

摘要: 深度神经网络(DNN)越来越多地应用于身份认证和自动驾驶等关键应用中，其中对对手攻击的健壮性至关重要。这些攻击可以利用微小的扰动来导致显著的预测误差，因此增强DNN的弹性是至关重要的。传统的防御方法通常依赖于获取详细的模型信息，这引发了隐私问题，因为模型所有者可能不愿分享此类数据。相比之下，现有的黑盒防御方法无法针对各种类型的对抗性攻击提供通用的防御。为了应对这些挑战，我们引入了DUCD，这是一种通用的黑盒防御方法，不需要访问目标模型的参数或体系结构。我们的方法包括通过使用数据查询目标模型来提取目标模型，在保护数据隐私的同时创建白盒代理。我们使用基于随机平滑和优化噪声选择的认证防御进一步增强了该代理模型，从而实现了对广泛的对抗性攻击的稳健防御。对代理模型和目标模型的认证防御的比较评估证明了我们方法的有效性。在多个图像分类数据集上的实验表明，DUCD不仅性能优于现有的黑盒防御，而且与白盒防御的精度相当，同时增强了数据隐私，降低了隶属度推理攻击的成功率。



## **27. Box-Free Model Watermarks Are Prone to Black-Box Removal Attacks**

无框模型水印容易受到黑匣子删除攻击 cs.CV

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2405.09863v3) [paper-pdf](http://arxiv.org/pdf/2405.09863v3)

**Authors**: Haonan An, Guang Hua, Zhiping Lin, Yuguang Fang

**Abstract**: Box-free model watermarking is an emerging technique to safeguard the intellectual property of deep learning models, particularly those for low-level image processing tasks. Existing works have verified and improved its effectiveness in several aspects. However, in this paper, we reveal that box-free model watermarking is prone to removal attacks, even under the real-world threat model such that the protected model and the watermark extractor are in black boxes. Under this setting, we carry out three studies. 1) We develop an extractor-gradient-guided (EGG) remover and show its effectiveness when the extractor uses ReLU activation only. 2) More generally, for an unknown extractor, we leverage adversarial attacks and design the EGG remover based on the estimated gradients. 3) Under the most stringent condition that the extractor is inaccessible, we design a transferable remover based on a set of private proxy models. In all cases, the proposed removers can successfully remove embedded watermarks while preserving the quality of the processed images, and we also demonstrate that the EGG remover can even replace the watermarks. Extensive experimental results verify the effectiveness and generalizability of the proposed attacks, revealing the vulnerabilities of the existing box-free methods and calling for further research.

摘要: 无盒模型水印是一种新兴的保护深度学习模型知识产权的技术，尤其是用于低层图像处理任务的模型。已有的工作在几个方面验证和改进了它的有效性。然而，在本文中，我们揭示了无盒模型水印容易受到移除攻击，即使在真实世界的威胁模型下，受保护的模型和水印抽取器都在黑盒中。在此背景下，我们开展了三个方面的研究。1)我们开发了一种萃取器-梯度引导(EGG)去除器，并在仅使用RELU激活的情况下展示了其有效性。2)更一般地，对于未知的提取者，我们利用对抗性攻击，并基于估计的梯度来设计鸡蛋去除器。3)在抽取器不可访问的最严格条件下，基于一组私有代理模型设计了一个可转移的抽取器。在所有情况下，所提出的去除器都可以在保持处理图像质量的情况下成功地去除嵌入的水印，并且我们还证明了鸡蛋去除器甚至可以替换水印。大量的实验结果验证了所提出的攻击方法的有效性和泛化能力，揭示了现有去盒方法的弱点，需要进一步研究。



## **28. Prompt-Agnostic Adversarial Perturbation for Customized Diffusion Models**

定制扩散模型的预算不可知对抗扰动 cs.CV

33 pages, 14 figures, under review

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10571v1) [paper-pdf](http://arxiv.org/pdf/2408.10571v1)

**Authors**: Cong Wan, Yuhang He, Xiang Song, Yihong Gong

**Abstract**: Diffusion models have revolutionized customized text-to-image generation, allowing for efficient synthesis of photos from personal data with textual descriptions. However, these advancements bring forth risks including privacy breaches and unauthorized replication of artworks. Previous researches primarily center around using prompt-specific methods to generate adversarial examples to protect personal images, yet the effectiveness of existing methods is hindered by constrained adaptability to different prompts. In this paper, we introduce a Prompt-Agnostic Adversarial Perturbation (PAP) method for customized diffusion models. PAP first models the prompt distribution using a Laplace Approximation, and then produces prompt-agnostic perturbations by maximizing a disturbance expectation based on the modeled distribution. This approach effectively tackles the prompt-agnostic attacks, leading to improved defense stability. Extensive experiments in face privacy and artistic style protection, demonstrate the superior generalization of our method in comparison to existing techniques.

摘要: 扩散模型彻底改变了定制的文本到图像的生成，允许从具有文本描述的个人数据高效地合成照片。然而，这些进步带来了包括侵犯隐私和未经授权复制艺术品在内的风险。以往的研究主要集中在使用特定于提示的方法来生成对抗性的例子来保护个人形象，然而现有方法的有效性受到对不同提示的限制适应性的阻碍。在这篇文章中，我们介绍了定制扩散模型的即时不可知对抗扰动(PAP)方法。PAP首先使用拉普拉斯近似对瞬发分布进行建模，然后基于建模的分布最大化扰动期望来产生与瞬发无关的扰动。这种方法有效地解决了即时不可知攻击，从而提高了防御稳定性。在人脸隐私和艺术风格保护方面的大量实验表明，与现有技术相比，该方法具有更好的泛化能力。



## **29. Enhancing Adversarial Transferability with Adversarial Weight Tuning**

通过对抗权重调整增强对抗可移植性 cs.CR

13 pages

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.09469v2) [paper-pdf](http://arxiv.org/pdf/2408.09469v2)

**Authors**: Jiahao Chen, Zhou Feng, Rui Zeng, Yuwen Pu, Chunyi Zhou, Yi Jiang, Yuyou Gan, Jinbao Li, Shouling Ji

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial examples (AEs) that mislead the model while appearing benign to human observers. A critical concern is the transferability of AEs, which enables black-box attacks without direct access to the target model. However, many previous attacks have failed to explain the intrinsic mechanism of adversarial transferability. In this paper, we rethink the property of transferable AEs and reformalize the formulation of transferability. Building on insights from this mechanism, we analyze the generalization of AEs across models with different architectures and prove that we can find a local perturbation to mitigate the gap between surrogate and target models. We further establish the inner connections between model smoothness and flat local maxima, both of which contribute to the transferability of AEs. Further, we propose a new adversarial attack algorithm, \textbf{A}dversarial \textbf{W}eight \textbf{T}uning (AWT), which adaptively adjusts the parameters of the surrogate model using generated AEs to optimize the flat local maxima and model smoothness simultaneously, without the need for extra data. AWT is a data-free tuning method that combines gradient-based and model-based attack methods to enhance the transferability of AEs. Extensive experiments on a variety of models with different architectures on ImageNet demonstrate that AWT yields superior performance over other attacks, with an average increase of nearly 5\% and 10\% attack success rates on CNN-based and Transformer-based models, respectively, compared to state-of-the-art attacks.

摘要: 深度神经网络(DNN)很容易受到敌意例子(AE)的攻击，这些例子误导了模型，同时对人类观察者来说是良性的。一个关键的问题是AEs的可转移性，这使得黑盒攻击能够在不直接访问目标模型的情况下进行。然而，以往的许多攻击都未能解释对抗性转移的内在机制。在本文中，我们重新思考了可转让实体的性质，并对可转让的提法进行了改造。在此机制的基础上，我们分析了不同体系结构模型之间的AEs泛化，并证明了我们可以找到局部扰动来缓解代理模型和目标模型之间的差距。我们进一步建立了模型光滑性与平坦局部极大值之间的内在联系，这两者都有助于AEs的可转移性。在此基础上，提出了一种新的对抗性攻击算法AWT是一种无数据调整方法，它结合了基于梯度和基于模型的攻击方法来增强AE的可转移性。在ImageNet上对不同体系结构的各种模型进行的大量实验表明，AWT的攻击性能优于其他攻击，基于CNN和基于Transformer的模型的攻击成功率比最先进的攻击分别提高了近5%和10%。



## **30. PromptBench: A Unified Library for Evaluation of Large Language Models**

EntBench：大型语言模型评估的统一库 cs.AI

Accepted by Journal of Machine Learning Research (JMLR); code:  https://github.com/microsoft/promptbench

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2312.07910v3) [paper-pdf](http://arxiv.org/pdf/2312.07910v3)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.

摘要: 大型语言模型（LLM）的评估对于评估其性能和降低潜在的安全风险至关重要。本文中，我们介绍了EmotiBench，这是一个评估LLM的统一图书馆。它由研究人员易于使用和扩展的几个关键组件组成：即时构建、即时工程、数据集和模型加载、对抗即时攻击、动态评估协议和分析工具。EntBench旨在成为一个开放、通用和灵活的代码库，用于研究目的，可以促进创建新基准、部署下游应用程序和设计新评估协议的原始研究。该代码可在https://github.com/microsoft/promptbench上获取，并将持续支持。



## **31. Fight Perturbations with Perturbations: Defending Adversarial Attacks via Neuron Influence**

用扰动对抗扰动：通过神经元影响防御对抗攻击 cs.CV

Final version. Accepted to IEEE Transactions on Dependable and Secure  Computing

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2112.13060v3) [paper-pdf](http://arxiv.org/pdf/2112.13060v3)

**Authors**: Ruoxi Chen, Haibo Jin, Haibin Zheng, Jinyin Chen, Zhenguang Liu

**Abstract**: The vulnerabilities of deep learning models towards adversarial attacks have attracted increasing attention, especially when models are deployed in security-critical domains. Numerous defense methods, including reactive and proactive ones, have been proposed for model robustness improvement. Reactive defenses, such as conducting transformations to remove perturbations, usually fail to handle large perturbations. The proactive defenses that involve retraining, suffer from the attack dependency and high computation cost. In this paper, we consider defense methods from the general effect of adversarial attacks that take on neurons inside the model. We introduce the concept of neuron influence, which can quantitatively measure neurons' contribution to correct classification. Then, we observe that almost all attacks fool the model by suppressing neurons with larger influence and enhancing those with smaller influence. Based on this, we propose \emph{Neuron-level Inverse Perturbation} (NIP), a novel defense against general adversarial attacks. It calculates neuron influence from benign examples and then modifies input examples by generating inverse perturbations that can in turn strengthen neurons with larger influence and weaken those with smaller influence.

摘要: 深度学习模型对敌意攻击的脆弱性引起了越来越多的关注，特别是当模型部署在安全关键领域时。为了提高模型的稳健性，人们提出了多种防御方法，包括被动防御和主动防御。被动防御，例如进行变换以消除扰动，通常无法处理大扰动。主动防御涉及再训练，存在攻击依赖性和计算代价高等问题。在本文中，我们从对抗性攻击对模型内部神经元的一般影响出发，考虑防御方法。我们引入了神经元影响的概念，它可以定量地衡量神经元对正确分类的贡献。然后，我们观察到几乎所有的攻击都通过抑制影响较大的神经元和增强影响较小的神经元来愚弄模型。在此基础上，我们提出了一种新的防御一般敌意攻击的方法--神经元水平逆摄动(NIP)。它从良性样本中计算神经元的影响，然后通过产生逆扰动来修改输入样本，反过来可以增强影响较大的神经元，削弱影响较小的神经元。



## **32. Detecting Adversarial Attacks in Semantic Segmentation via Uncertainty Estimation: A Deep Analysis**

通过不确定性估计检测语义分割中的对抗性攻击：深入分析 cs.CV

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10021v1) [paper-pdf](http://arxiv.org/pdf/2408.10021v1)

**Authors**: Kira Maag, Roman Resner, Asja Fischer

**Abstract**: Deep neural networks have demonstrated remarkable effectiveness across a wide range of tasks such as semantic segmentation. Nevertheless, these networks are vulnerable to adversarial attacks that add imperceptible perturbations to the input image, leading to false predictions. This vulnerability is particularly dangerous in safety-critical applications like automated driving. While adversarial examples and defense strategies are well-researched in the context of image classification, there is comparatively less research focused on semantic segmentation. Recently, we have proposed an uncertainty-based method for detecting adversarial attacks on neural networks for semantic segmentation. We observed that uncertainty, as measured by the entropy of the output distribution, behaves differently on clean versus adversely perturbed images, and we utilize this property to differentiate between the two. In this extended version of our work, we conduct a detailed analysis of uncertainty-based detection of adversarial attacks including a diverse set of adversarial attacks and various state-of-the-art neural networks. Our numerical experiments show the effectiveness of the proposed uncertainty-based detection method, which is lightweight and operates as a post-processing step, i.e., no model modifications or knowledge of the adversarial example generation process are required.

摘要: 深度神经网络在语义分割等一系列任务中表现出了显著的有效性。然而，这些网络很容易受到敌意攻击，这些攻击会给输入图像添加不可察觉的扰动，导致错误预测。该漏洞在自动驾驶等安全关键型应用程序中尤其危险。虽然在图像分类的背景下，对抗性例子和防御策略已经得到了很好的研究，但针对语义分割的研究相对较少。最近，我们提出了一种基于不确定性的神经网络敌意攻击检测方法，用于语义分割。我们观察到，通过输出分布的熵来衡量不确定性，在干净的图像和反向扰动的图像上表现出不同的行为，我们利用这一特性来区分两者。在我们工作的这个扩展版本中，我们对基于不确定性的对抗性攻击检测进行了详细的分析，包括一组不同的对抗性攻击和各种最先进的神经网络。我们的数值实验表明了基于不确定性的检测方法的有效性，该方法是轻量级的，并且作为后处理步骤进行操作，即不需要修改模型或了解对抗性示例的生成过程。



## **33. Adversarial Prompt Tuning for Vision-Language Models**

视觉语言模型的对抗性即时调优 cs.CV

ECCV 2024

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2311.11261v3) [paper-pdf](http://arxiv.org/pdf/2311.11261v3)

**Authors**: Jiaming Zhang, Xingjun Ma, Xin Wang, Lingyu Qiu, Jiaqi Wang, Yu-Gang Jiang, Jitao Sang

**Abstract**: With the rapid advancement of multimodal learning, pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capacities in bridging the gap between visual and language modalities. However, these models remain vulnerable to adversarial attacks, particularly in the image modality, presenting considerable security risks. This paper introduces Adversarial Prompt Tuning (AdvPT), a novel technique to enhance the adversarial robustness of image encoders in VLMs. AdvPT innovatively leverages learnable text prompts and aligns them with adversarial image embeddings, to address the vulnerabilities inherent in VLMs without the need for extensive parameter training or modification of the model architecture. We demonstrate that AdvPT improves resistance against white-box and black-box adversarial attacks and exhibits a synergistic effect when combined with existing image-processing-based defense techniques, further boosting defensive capabilities. Comprehensive experimental analyses provide insights into adversarial prompt tuning, a novel paradigm devoted to improving resistance to adversarial images through textual input modifications, paving the way for future robust multimodal learning research. These findings open up new possibilities for enhancing the security of VLMs. Our code is available at https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning.

摘要: 随着多通道学习的快速发展，诸如CLIP等预先训练的视觉语言模型在弥合视觉和语言通道之间的差距方面显示出了显著的能力。然而，这些模型仍然容易受到敌意攻击，特别是在图像模式方面，这带来了相当大的安全风险。本文介绍了对抗性提示调优(AdvPT)技术，这是一种在VLMS中增强图像编码器对抗性稳健性的新技术。AdvPT创新性地利用可学习的文本提示，并将其与对抗性图像嵌入相结合，以解决VLM中固有的漏洞，而无需进行广泛的参数培训或修改模型体系结构。我们证明，AdvPT提高了对白盒和黑盒攻击的抵抗力，并与现有的基于图像处理的防御技术相结合，显示出协同效应，进一步增强了防御能力。全面的实验分析提供了对对抗性即时调整的见解，这是一种致力于通过修改文本输入来提高对对抗性图像的抵抗力的新范式，为未来稳健的多通道学习研究铺平了道路。这些发现为增强VLM的安全性开辟了新的可能性。我们的代码可以在https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning.上找到



## **34. Segment-Anything Models Achieve Zero-shot Robustness in Autonomous Driving**

分段任意模型在自动驾驶中实现零攻击鲁棒性 cs.CV

Accepted to IAVVC 2024

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.09839v1) [paper-pdf](http://arxiv.org/pdf/2408.09839v1)

**Authors**: Jun Yan, Pengyu Wang, Danni Wang, Weiquan Huang, Daniel Watzenig, Huilin Yin

**Abstract**: Semantic segmentation is a significant perception task in autonomous driving. It suffers from the risks of adversarial examples. In the past few years, deep learning has gradually transitioned from convolutional neural network (CNN) models with a relatively small number of parameters to foundation models with a huge number of parameters. The segment-anything model (SAM) is a generalized image segmentation framework that is capable of handling various types of images and is able to recognize and segment arbitrary objects in an image without the need to train on a specific object. It is a unified model that can handle diverse downstream tasks, including semantic segmentation, object detection, and tracking. In the task of semantic segmentation for autonomous driving, it is significant to study the zero-shot adversarial robustness of SAM. Therefore, we deliver a systematic empirical study on the robustness of SAM without additional training. Based on the experimental results, the zero-shot adversarial robustness of the SAM under the black-box corruptions and white-box adversarial attacks is acceptable, even without the need for additional training. The finding of this study is insightful in that the gigantic model parameters and huge amounts of training data lead to the phenomenon of emergence, which builds a guarantee of adversarial robustness. SAM is a vision foundation model that can be regarded as an early prototype of an artificial general intelligence (AGI) pipeline. In such a pipeline, a unified model can handle diverse tasks. Therefore, this research not only inspects the impact of vision foundation models on safe autonomous driving but also provides a perspective on developing trustworthy AGI. The code is available at: https://github.com/momo1986/robust_sam_iv.

摘要: 语义分割是自动驾驶中一项重要的感知任务。它面临着对抗性例子的风险。在过去的几年里，深度学习逐渐从参数相对较少的卷积神经网络(CNN)模型过渡到参数数量巨大的基础模型。任意分割模型(SAM)是一个通用的图像分割框架，它能够处理各种类型的图像，并且能够识别和分割图像中的任意对象，而不需要对特定对象进行训练。它是一个统一的模型，可以处理不同的下游任务，包括语义分割、目标检测和跟踪。在自主驾驶的语义分割任务中，研究SAM的零射击对抗健壮性具有重要意义。因此，我们在没有额外训练的情况下对SAM的稳健性进行了系统的实证研究。基于实验结果，即使不需要额外的训练，SAM在黑盒腐败和白盒对抗攻击下的零射击对抗健壮性也是可以接受的。这项研究的发现是有洞察力的，因为庞大的模型参数和大量的训练数据导致了涌现现象，这为对手的稳健性提供了保证。SAM是一个视觉基础模型，可以被视为人工通用智能(AGI)管道的早期原型。在这样的管道中，统一的模型可以处理不同的任务。因此，本研究不仅考察了视觉基础模型对安全自动驾驶的影响，而且为开发可信赖的自动驾驶系统提供了一个视角。代码可从以下网址获得：https://github.com/momo1986/robust_sam_iv.



## **35. Patch of Invisibility: Naturalistic Physical Black-Box Adversarial Attacks on Object Detectors**

隐形补丁：对物体检测器的自然主义物理黑匣子对抗攻击 cs.CV

Accepted at MLCS @ ECML-PKDD 2024

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2303.04238v5) [paper-pdf](http://arxiv.org/pdf/2303.04238v5)

**Authors**: Raz Lapid, Eylon Mizrahi, Moshe Sipper

**Abstract**: Adversarial attacks on deep-learning models have been receiving increased attention in recent years. Work in this area has mostly focused on gradient-based techniques, so-called "white-box" attacks, wherein the attacker has access to the targeted model's internal parameters; such an assumption is usually unrealistic in the real world. Some attacks additionally use the entire pixel space to fool a given model, which is neither practical nor physical (i.e., real-world). On the contrary, we propose herein a direct, black-box, gradient-free method that uses the learned image manifold of a pretrained generative adversarial network (GAN) to generate naturalistic physical adversarial patches for object detectors. To our knowledge this is the first and only method that performs black-box physical attacks directly on object-detection models, which results with a model-agnostic attack. We show that our proposed method works both digitally and physically. We compared our approach against four different black-box attacks with different configurations. Our approach outperformed all other approaches that were tested in our experiments by a large margin.

摘要: 近年来，针对深度学习模型的对抗性攻击受到越来越多的关注。这一领域的工作主要集中在基于梯度的技术上，即所谓的“白盒”攻击，即攻击者可以访问目标模型的内部参数；这种假设在现实世界中通常是不现实的。一些攻击还使用整个像素空间来愚弄给定的模型，这既不实用也不物理(即，现实世界)。相反，我们在这里提出了一种直接的、黑盒的、无梯度的方法，该方法使用预先训练的生成性对抗网络(GAN)的学习图像流形来为目标检测器生成自然的物理对抗斑块。据我们所知，这是第一种也是唯一一种直接对目标检测模型执行黑盒物理攻击的方法，这导致了与模型无关的攻击。我们证明了我们提出的方法在数字和物理上都是有效的。我们将我们的方法与四种不同配置的不同黑盒攻击进行了比较。我们的方法远远超过了在我们的实验中测试的所有其他方法。



## **36. Regularization for Adversarial Robust Learning**

对抗鲁棒学习的正规化 cs.LG

51 pages, 5 figures

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.09672v1) [paper-pdf](http://arxiv.org/pdf/2408.09672v1)

**Authors**: Jie Wang, Rui Gao, Yao Xie

**Abstract**: Despite the growing prevalence of artificial neural networks in real-world applications, their vulnerability to adversarial attacks remains to be a significant concern, which motivates us to investigate the robustness of machine learning models. While various heuristics aim to optimize the distributionally robust risk using the $\infty$-Wasserstein metric, such a notion of robustness frequently encounters computation intractability. To tackle the computational challenge, we develop a novel approach to adversarial training that integrates $\phi$-divergence regularization into the distributionally robust risk function. This regularization brings a notable improvement in computation compared with the original formulation. We develop stochastic gradient methods with biased oracles to solve this problem efficiently, achieving the near-optimal sample complexity. Moreover, we establish its regularization effects and demonstrate it is asymptotic equivalence to a regularized empirical risk minimization (ERM) framework, by considering various scaling regimes of the regularization parameter $\eta$ and robustness level $\rho$. These regimes yield gradient norm regularization, variance regularization, or a smoothed gradient norm regularization that interpolates between these extremes. We numerically validate our proposed method in supervised learning, reinforcement learning, and contextual learning and showcase its state-of-the-art performance against various adversarial attacks.

摘要: 尽管人工神经网络在现实世界中的应用越来越普遍，但它们对对手攻击的脆弱性仍然是一个重要的问题，这促使我们研究机器学习模型的健壮性。虽然各种启发式方法的目标是使用$\infty$-Wasserstein度量来优化分布健壮性风险，但这样的健壮性概念经常遇到计算困难。为了解决计算上的挑战，我们开发了一种新的对抗性训练方法，将$Phi$-发散正则化整合到分布稳健的风险函数中。与原公式相比，这种正则化方法在计算上有了显著的改进。我们发展了带有有偏预言的随机梯度方法来有效地解决这一问题，获得了接近最优的样本复杂度。此外，我们建立了它的正则化效应，并证明了它与正则化经验风险最小化(ERM)框架的渐近等价，通过考虑正则化参数和稳健性水平的不同标度机制。这些区域产生在这些极值之间内插的梯度范数正则化、方差正则化或平滑的梯度范数正则化。我们在监督学习、强化学习和上下文学习中对我们提出的方法进行了数值验证，并展示了它在抵抗各种对手攻击方面的最新表现。



## **37. Symbiotic Game and Foundation Models for Cyber Deception Operations in Strategic Cyber Warfare**

战略网络战中网络欺骗行动的共生博弈和基础模型 cs.CR

40 pages, 7 figures, 2 tables

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2403.10570v2) [paper-pdf](http://arxiv.org/pdf/2403.10570v2)

**Authors**: Tao Li, Quanyan Zhu

**Abstract**: We are currently facing unprecedented cyber warfare with the rapid evolution of tactics, increasing asymmetry of intelligence, and the growing accessibility of hacking tools. In this landscape, cyber deception emerges as a critical component of our defense strategy against increasingly sophisticated attacks. This chapter aims to highlight the pivotal role of game-theoretic models and foundation models (FMs) in analyzing, designing, and implementing cyber deception tactics. Game models (GMs) serve as a foundational framework for modeling diverse adversarial interactions, allowing us to encapsulate both adversarial knowledge and domain-specific insights. Meanwhile, FMs serve as the building blocks for creating tailored machine learning models suited to given applications. By leveraging the synergy between GMs and FMs, we can advance proactive and automated cyber defense mechanisms by not only securing our networks against attacks but also enhancing their resilience against well-planned operations. This chapter discusses the games at the tactical, operational, and strategic levels of warfare, delves into the symbiotic relationship between these methodologies, and explores relevant applications where such a framework can make a substantial impact in cybersecurity. The chapter discusses the promising direction of the multi-agent neurosymbolic conjectural learning (MANSCOL), which allows the defender to predict adversarial behaviors, design adaptive defensive deception tactics, and synthesize knowledge for the operational level synthesis and adaptation. FMs serve as pivotal tools across various functions for MANSCOL, including reinforcement learning, knowledge assimilation, formation of conjectures, and contextual representation. This chapter concludes with a discussion of the challenges associated with FMs and their application in the domain of cybersecurity.

摘要: 我们目前正面临着前所未有的网络战，战术的快速演变，情报的日益不对称，以及黑客工具的日益普及。在这种情况下，网络欺骗成为我们抵御日益复杂的攻击的防御战略的关键组成部分。本章旨在强调博弈论模型和基础模型(FM)在分析、设计和实施网络欺骗战术中的关键作用。游戏模型(GM)作为建模各种对抗性交互的基本框架，允许我们封装对抗性知识和特定领域的见解。同时，FM作为创建适合特定应用的定制机器学习模型的构建块。通过利用GM和FM之间的协同作用，我们可以推进主动和自动化的网络防御机制，不仅确保我们的网络免受攻击，而且增强其对精心规划的行动的弹性。本章讨论了战争的战术、作战和战略层面的游戏，深入探讨了这些方法之间的共生关系，并探索了此类框架可以在网络安全中产生重大影响的相关应用。本章讨论了多智能体神经符号猜想学习(MANSCOL)的发展方向，它允许防御者预测对手的行为，设计自适应的防御性欺骗策略，并为作战级别的综合和适应综合知识。FMS是MANSCOL的各种功能的关键工具，包括强化学习、知识同化、猜想的形成和上下文表示。本章最后讨论了与FMS相关的挑战及其在网络安全领域的应用。



## **38. Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework**

Bergeron：通过基于意识的一致框架打击敌对攻击 cs.CR

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2312.00029v3) [paper-pdf](http://arxiv.org/pdf/2312.00029v3)

**Authors**: Matthew Pisano, Peter Ly, Abraham Sanders, Bingsheng Yao, Dakuo Wang, Tomek Strzalkowski, Mei Si

**Abstract**: Research into AI alignment has grown considerably since the recent introduction of increasingly capable Large Language Models (LLMs). Unfortunately, modern methods of alignment still fail to fully prevent harmful responses when models are deliberately attacked. Such vulnerabilities can lead to LLMs being manipulated into generating hazardous content: from instructions for creating dangerous materials to inciting violence or endorsing unethical behaviors. To help mitigate this issue, we introduce Bergeron: a framework designed to improve the robustness of LLMs against attacks without any additional parameter fine-tuning. Bergeron is organized into two tiers; with a secondary LLM acting as a guardian to the primary LLM. This framework better safeguards the primary model against incoming attacks while monitoring its output for any harmful content. Empirical analysis reviews that by using Bergeron to complement models with existing alignment training, we can significantly improve the robustness and safety of multiple, commonly used commercial and open-source LLMs. Specifically, we found that models integrated with Bergeron are, on average, nearly seven times more resistant to attacks compared to models without such support.

摘要: 自从最近引入了功能越来越强大的大型语言模型(LLM)以来，对人工智能对齐的研究有了很大的增长。不幸的是，现代的校准方法仍然不能完全防止模型受到故意攻击时的有害反应。这些漏洞可能导致LLMS被操纵来生成危险内容：从创建危险材料的说明到煽动暴力或支持不道德行为。为了帮助缓解这个问题，我们引入了Bergeron：一个旨在提高LLM抵御攻击的健壮性的框架，而不需要任何额外的参数微调。Bergeron被组织成两级；辅助LLM充当主要LLM的监护人。此框架可以更好地保护主要模型免受来袭攻击，同时监控其输出中是否有任何有害内容。经验分析认为，通过使用Bergeron来补充模型与现有的比对训练，我们可以显著提高多个常用的商业和开源LLM的稳健性和安全性。具体地说，我们发现，与没有这种支持的型号相比，集成了Bergeron的型号平均抵抗攻击的能力要高出近7倍。



## **39. Interpreting Global Perturbation Robustness of Image Models using Axiomatic Spectral Importance Decomposition**

使用公理谱重要性分解解释图像模型的全局扰动鲁棒性 cs.AI

Accepted by Transactions on Machine Learning Research (TMLR 2024)

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.01139v2) [paper-pdf](http://arxiv.org/pdf/2408.01139v2)

**Authors**: Róisín Luo, James McDermott, Colm O'Riordan

**Abstract**: Perturbation robustness evaluates the vulnerabilities of models, arising from a variety of perturbations, such as data corruptions and adversarial attacks. Understanding the mechanisms of perturbation robustness is critical for global interpretability. We present a model-agnostic, global mechanistic interpretability method to interpret the perturbation robustness of image models. This research is motivated by two key aspects. First, previous global interpretability works, in tandem with robustness benchmarks, e.g. mean corruption error (mCE), are not designed to directly interpret the mechanisms of perturbation robustness within image models. Second, we notice that the spectral signal-to-noise ratios (SNR) of perturbed natural images exponentially decay over the frequency. This power-law-like decay implies that: Low-frequency signals are generally more robust than high-frequency signals -- yet high classification accuracy can not be achieved by low-frequency signals alone. By applying Shapley value theory, our method axiomatically quantifies the predictive powers of robust features and non-robust features within an information theory framework. Our method, dubbed as \textbf{I-ASIDE} (\textbf{I}mage \textbf{A}xiomatic \textbf{S}pectral \textbf{I}mportance \textbf{D}ecomposition \textbf{E}xplanation), provides a unique insight into model robustness mechanisms. We conduct extensive experiments over a variety of vision models pre-trained on ImageNet to show that \textbf{I-ASIDE} can not only \textbf{measure} the perturbation robustness but also \textbf{provide interpretations} of its mechanisms.

摘要: 扰动稳健性评估由各种扰动引起的模型的脆弱性，例如数据损坏和对抗性攻击。理解扰动稳健性的机制对于全局可解释性至关重要。我们提出了一种模型不可知的全局机械可解释性方法来解释图像模型的扰动稳健性。这项研究的动机有两个关键方面。首先，以前的全局可解释性与稳健性基准一起工作，例如平均破坏误差(MCE)，不是被设计成直接解释图像模型中的扰动稳健性的机制。其次，我们注意到受扰动的自然图像的光谱信噪比(SNR)随频率呈指数衰减。这种类似幂规律的衰减意味着：低频信号通常比高频信号更健壮--然而，仅靠低频信号不能达到高分类精度。通过应用Shapley值理论，我们的方法在信息论框架内公理地量化了稳健特征和非稳健特征的预测能力。我们的方法称为Textbf{i-side}(Textbf{I}MAGE\Textbf{A}X-Ait\Textbf{S}频谱\Textbf{I}M重要\Textbf{D}分解\Textbf{E}解释)，提供了对模型健壮性机制的独特见解。我们在ImageNet上预先训练的各种视觉模型上进行了大量的实验，结果表明，文本bf{i-side}不仅可以测量扰动的稳健性，而且可以对其机制进行解释。



## **40. WPN: An Unlearning Method Based on N-pair Contrastive Learning in Language Models**

WPN：一种基于语言模型N对对比学习的去学习方法 cs.CL

ECAI 2024

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09459v1) [paper-pdf](http://arxiv.org/pdf/2408.09459v1)

**Authors**: Guitao Chen, Yunshen Wang, Hongye Sun, Guang Chen

**Abstract**: Generative language models (LMs) offer numerous advantages but may produce inappropriate or harmful outputs due to the harmful knowledge acquired during pre-training. This knowledge often manifests as undesirable correspondences, such as "harmful prompts" leading to "harmful outputs," which our research aims to mitigate through unlearning techniques.However, existing unlearning methods based on gradient ascent can significantly impair the performance of LMs. To address this issue, we propose a novel approach called Weighted Positional N-pair (WPN) Learning, which leverages position-weighted mean pooling within an n-pair contrastive learning framework. WPN is designed to modify the output distribution of LMs by eliminating specific harmful outputs (e.g., replacing toxic responses with neutral ones), thereby transforming the model's behavior from "harmful prompt-harmful output" to "harmful prompt-harmless response".Experiments on OPT and GPT-NEO LMs show that WPN effectively reduces the proportion of harmful responses, achieving a harmless rate of up to 95.8\% while maintaining stable performance on nine common benchmarks (with less than 2\% degradation on average). Moreover, we provide empirical evidence to demonstrate WPN's ability to weaken the harmful correspondences in terms of generalizability and robustness, as evaluated on out-of-distribution test sets and under adversarial attacks.

摘要: 生成语言模型(LMS)有许多优点，但可能会产生不适当或有害的输出，因为在预培训期间获得了有害的知识。这些知识经常表现为不希望看到的对应关系，例如“有害提示”导致“有害输出”，我们的研究旨在通过遗忘技术来缓解这种情况。然而，现有的基于梯度上升的遗忘方法会显著影响LMS的性能。为了解决这个问题，我们提出了一种新的方法，称为加权位置N对(WPN)学习，它利用n对对比学习框架中的位置加权平均池。在OPT和GPT-neo LMS上的实验表明，WPN有效地减少了有害响应的比例，在9个常用基准上保持了稳定的性能(平均降级小于2)，从而改变了LMS的输出分布。此外，我们提供了经验证据来证明WPN在泛化能力和稳健性方面能够削弱有害的对应关系，如在分布外测试集上和在敌意攻击下的评估。



## **41. XAI-Based Detection of Adversarial Attacks on Deepfake Detectors**

基于XAI检测Deepfake检测器上的对抗攻击 cs.CR

Accepted at TMLR 2024

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2403.02955v2) [paper-pdf](http://arxiv.org/pdf/2403.02955v2)

**Authors**: Ben Pinhasov, Raz Lapid, Rony Ohayon, Moshe Sipper, Yehudit Aperstein

**Abstract**: We introduce a novel methodology for identifying adversarial attacks on deepfake detectors using eXplainable Artificial Intelligence (XAI). In an era characterized by digital advancement, deepfakes have emerged as a potent tool, creating a demand for efficient detection systems. However, these systems are frequently targeted by adversarial attacks that inhibit their performance. We address this gap, developing a defensible deepfake detector by leveraging the power of XAI. The proposed methodology uses XAI to generate interpretability maps for a given method, providing explicit visualizations of decision-making factors within the AI models. We subsequently employ a pretrained feature extractor that processes both the input image and its corresponding XAI image. The feature embeddings extracted from this process are then used for training a simple yet effective classifier. Our approach contributes not only to the detection of deepfakes but also enhances the understanding of possible adversarial attacks, pinpointing potential vulnerabilities. Furthermore, this approach does not change the performance of the deepfake detector. The paper demonstrates promising results suggesting a potential pathway for future deepfake detection mechanisms. We believe this study will serve as a valuable contribution to the community, sparking much-needed discourse on safeguarding deepfake detectors.

摘要: 我们介绍了一种利用可解释人工智能(XAI)来识别针对深度假冒检测器的对抗性攻击的新方法。在一个以数字进步为特征的时代，深度假冒已经成为一种强有力的工具，创造了对高效检测系统的需求。然而，这些系统经常成为抑制其性能的对抗性攻击的目标。我们解决了这个问题，通过利用XAI的能力开发了一个可防御的深度伪检测器。所提出的方法使用XAI为给定的方法生成可解释性地图，提供人工智能模型中决策因素的显式可视化。随后，我们采用了一个预先训练的特征抽取器来处理输入图像及其对应的XAI图像。然后使用从该过程中提取的特征嵌入来训练简单而有效的分类器。我们的方法不仅有助于深度假冒的检测，还有助于增强对可能的敌意攻击的理解，准确地定位潜在的漏洞。此外，该方法不会改变深度伪检测器的性能。这篇论文展示了令人振奋的结果，为未来的深度伪检测机制提供了一条潜在的途径。我们相信，这项研究将对社区做出有价值的贡献，引发关于保护深度假冒探测器的迫切需要的讨论。



## **42. Adversarial Attacked Teacher for Unsupervised Domain Adaptive Object Detection**

无监督领域自适应对象检测的对抗攻击教师 cs.CV

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09431v1) [paper-pdf](http://arxiv.org/pdf/2408.09431v1)

**Authors**: Kaiwen Wang, Yinzhe Shen, Martin Lauer

**Abstract**: Object detectors encounter challenges in handling domain shifts. Cutting-edge domain adaptive object detection methods use the teacher-student framework and domain adversarial learning to generate domain-invariant pseudo-labels for self-training. However, the pseudo-labels generated by the teacher model tend to be biased towards the majority class and often mistakenly include overconfident false positives and underconfident false negatives. We reveal that pseudo-labels vulnerable to adversarial attacks are more likely to be low-quality. To address this, we propose a simple yet effective framework named Adversarial Attacked Teacher (AAT) to improve the quality of pseudo-labels. Specifically, we apply adversarial attacks to the teacher model, prompting it to generate adversarial pseudo-labels to correct bias, suppress overconfidence, and encourage underconfident proposals. An adaptive pseudo-label regularization is introduced to emphasize the influence of pseudo-labels with high certainty and reduce the negative impacts of uncertain predictions. Moreover, robust minority objects verified by pseudo-label regularization are oversampled to minimize dataset imbalance without introducing false positives. Extensive experiments conducted on various datasets demonstrate that AAT achieves superior performance, reaching 52.6 mAP on Clipart1k, surpassing the previous state-of-the-art by 6.7%.

摘要: 物体探测器在处理区域移位时遇到了挑战。前沿领域自适应目标检测方法使用师生框架和领域对抗学习来生成领域不变的伪标签，用于自我训练。然而，教师模式生成的伪标签往往偏向于大多数班级，并且经常错误地包括过度自信的假阳性和不足自信的假阴性。我们发现，易受对手攻击的伪标签更有可能是低质量的。为了解决这个问题，我们提出了一个简单而有效的框架，称为对抗性攻击教师(AAT)，以提高伪标签的质量。具体地说，我们对教师模型应用对抗性攻击，促使它生成对抗性伪标签来纠正偏见，抑制过度自信，并鼓励信心不足的建议。引入了一种自适应伪标签正则化方法，以突出高确定性伪标签的影响，减少不确定预测带来的负面影响。此外，通过伪标签正则化验证的健壮少数对象被过采样，以最小化数据集的不平衡而不会引入误报。在不同的数据集上进行的广泛实验表明，AAT取得了优越的性能，在Clipart1k上达到了52.6MAP，超过了以前的最先进水平6.7%。



## **43. Rethinking Impersonation and Dodging Attacks on Face Recognition Systems**

重新思考模仿并躲避对人脸识别系统的攻击 cs.CV

Accepted to ACM MM 2024

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2401.08903v4) [paper-pdf](http://arxiv.org/pdf/2401.08903v4)

**Authors**: Fengfan Zhou, Qianyu Zhou, Bangjie Yin, Hui Zheng, Xuequan Lu, Lizhuang Ma, Hefei Ling

**Abstract**: Face Recognition (FR) systems can be easily deceived by adversarial examples that manipulate benign face images through imperceptible perturbations. Adversarial attacks on FR encompass two types: impersonation (targeted) attacks and dodging (untargeted) attacks. Previous methods often achieve a successful impersonation attack on FR, however, it does not necessarily guarantee a successful dodging attack on FR in the black-box setting. In this paper, our key insight is that the generation of adversarial examples should perform both impersonation and dodging attacks simultaneously. To this end, we propose a novel attack method termed as Adversarial Pruning (Adv-Pruning), to fine-tune existing adversarial examples to enhance their dodging capabilities while preserving their impersonation capabilities. Adv-Pruning consists of Priming, Pruning, and Restoration stages. Concretely, we propose Adversarial Priority Quantification to measure the region-wise priority of original adversarial perturbations, identifying and releasing those with minimal impact on absolute model output variances. Then, Biased Gradient Adaptation is presented to adapt the adversarial examples to traverse the decision boundaries of both the attacker and victim by adding perturbations favoring dodging attacks on the vacated regions, preserving the prioritized features of the original perturbations while boosting dodging performance. As a result, we can maintain the impersonation capabilities of original adversarial examples while effectively enhancing dodging capabilities. Comprehensive experiments demonstrate the superiority of our method compared with state-of-the-art adversarial attack methods.

摘要: 人脸识别(FR)系统很容易被敌意的例子欺骗，这些例子通过潜移默化的扰动来操纵良性的人脸图像。对FR的敌意攻击包括两种类型：模仿(目标)攻击和躲避(非目标)攻击。以往的方法往往能成功地实现对FR的模仿攻击，但在黑盒环境下，这并不一定能保证对FR的成功躲避攻击。在本文中，我们的主要观点是，生成敌意示例应该同时执行模仿攻击和躲避攻击。为此，我们提出了一种新的攻击方法，称为对抗性剪枝(ADV-Puning)，对现有的对抗性实例进行微调，以增强它们的躲避能力，同时保持它们的模拟能力。高级修剪包括启动、修剪和恢复三个阶段。具体地说，我们提出了对抗性优先级量化来度量原始对抗性扰动的区域优先级，识别并释放那些对绝对模型输出方差影响最小的扰动。然后，通过在空闲区域上添加有利于躲避攻击的扰动，保留了原始扰动的优先特征，同时提高了躲避性能，提出了有偏梯度自适应算法，使敌意例子能够穿越攻击者和受害者的决策边界。因此，我们可以在保持原始对抗性例子的模拟能力的同时，有效地增强躲避能力。综合实验表明，与现有的对抗性攻击方法相比，该方法具有一定的优越性。



## **44. Malacopula: adversarial automatic speaker verification attacks using a neural-based generalised Hammerstein model**

Malacopula：使用基于神经的广义Hammerstein模型的对抗性自动说话人验证攻击 eess.AS

Accepted at ASVspoof Workshop 2024

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09300v1) [paper-pdf](http://arxiv.org/pdf/2408.09300v1)

**Authors**: Massimiliano Todisco, Michele Panariello, Xin Wang, Héctor Delgado, Kong Aik Lee, Nicholas Evans

**Abstract**: We present Malacopula, a neural-based generalised Hammerstein model designed to introduce adversarial perturbations to spoofed speech utterances so that they better deceive automatic speaker verification (ASV) systems. Using non-linear processes to modify speech utterances, Malacopula enhances the effectiveness of spoofing attacks. The model comprises parallel branches of polynomial functions followed by linear time-invariant filters. The adversarial optimisation procedure acts to minimise the cosine distance between speaker embeddings extracted from spoofed and bona fide utterances. Experiments, performed using three recent ASV systems and the ASVspoof 2019 dataset, show that Malacopula increases vulnerabilities by a substantial margin. However, speech quality is reduced and attacks can be detected effectively under controlled conditions. The findings emphasise the need to identify new vulnerabilities and design defences to protect ASV systems from adversarial attacks in the wild.

摘要: 我们提出了Malacopula，这是一种基于神经的广义Hammerstein模型，旨在向欺骗的语音话语引入对抗性扰动，以便它们更好地欺骗自动说话人验证（ASV）系统。Malacopula使用非线性过程来修改语音话语，增强了欺骗攻击的有效性。该模型由多项函数的并行分支组成，后面是线性时不变过滤器。对抗优化过程的作用是最小化从欺骗和真实话语中提取的说话者嵌入之间的cos距离。使用三个最近的ASV系统和ASVspoof 2019数据集进行的实验表明，Malacopula大幅增加了漏洞。然而，语音质量会降低，并且可以在受控条件下有效检测攻击。研究结果强调，需要识别新的漏洞并设计防御措施，以保护ASV系统免受野外对抗攻击。



## **45. PADetBench: Towards Benchmarking Physical Attacks against Object Detection**

PADetBench：针对对象检测的物理攻击基准 cs.CV

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09181v1) [paper-pdf](http://arxiv.org/pdf/2408.09181v1)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Lap-Pui Chau, Shaohui Mei

**Abstract**: Physical attacks against object detection have gained increasing attention due to their significant practical implications. However, conducting physical experiments is extremely time-consuming and labor-intensive. Moreover, physical dynamics and cross-domain transformation are challenging to strictly regulate in the real world, leading to unaligned evaluation and comparison, severely hindering the development of physically robust models. To accommodate these challenges, we explore utilizing realistic simulation to thoroughly and rigorously benchmark physical attacks with fairness under controlled physical dynamics and cross-domain transformation. This resolves the problem of capturing identical adversarial images that cannot be achieved in the real world. Our benchmark includes 20 physical attack methods, 48 object detectors, comprehensive physical dynamics, and evaluation metrics. We also provide end-to-end pipelines for dataset generation, detection, evaluation, and further analysis. In addition, we perform 8064 groups of evaluation based on our benchmark, which includes both overall evaluation and further detailed ablation studies for controlled physical dynamics. Through these experiments, we provide in-depth analyses of physical attack performance and physical adversarial robustness, draw valuable observations, and discuss potential directions for future research.   Codebase: https://github.com/JiaweiLian/Benchmarking_Physical_Attack

摘要: 针对目标检测的物理攻击由于其重要的实际意义而受到越来越多的关注。然而，进行物理实验是极其耗时和劳动密集型的。此外，物理动力学和跨域转换在现实世界中面临严格规范的挑战，导致不一致的评估和比较，严重阻碍了物理稳健模型的发展。为了应对这些挑战，我们探索利用真实的模拟在受控的物理动力学和跨域转换下，彻底和严格地基准具有公平性的物理攻击。这解决了在现实世界中无法实现的捕获相同的对抗性图像的问题。我们的基准包括20种物理攻击方法、48个对象探测器、全面的物理动力学和评估指标。我们还提供用于数据集生成、检测、评估和进一步分析的端到端管道。此外，我们根据我们的基准进行了8064组评估，其中包括对受控物理动力学的整体评估和进一步的详细消融研究。通过这些实验，我们对物理攻击性能和物理对抗健壮性进行了深入的分析，得出了有价值的观察结果，并讨论了未来研究的潜在方向。码基：https://github.com/JiaweiLian/Benchmarking_Physical_Attack



## **46. Training Verifiably Robust Agents Using Set-Based Reinforcement Learning**

使用基于集的强化学习训练可验证稳健的代理 cs.LG

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09112v1) [paper-pdf](http://arxiv.org/pdf/2408.09112v1)

**Authors**: Manuel Wendl, Lukas Koller, Tobias Ladner, Matthias Althoff

**Abstract**: Reinforcement learning often uses neural networks to solve complex control tasks. However, neural networks are sensitive to input perturbations, which makes their deployment in safety-critical environments challenging. This work lifts recent results from formally verifying neural networks against such disturbances to reinforcement learning in continuous state and action spaces using reachability analysis. While previous work mainly focuses on adversarial attacks for robust reinforcement learning, we train neural networks utilizing entire sets of perturbed inputs and maximize the worst-case reward. The obtained agents are verifiably more robust than agents obtained by related work, making them more applicable in safety-critical environments. This is demonstrated with an extensive empirical evaluation of four different benchmarks.

摘要: 强化学习通常使用神经网络来解决复杂的控制任务。然而，神经网络对输入扰动很敏感，这使得它们在安全关键环境中的部署具有挑战性。这项工作将正式验证神经网络对抗此类干扰的最新结果提升到使用可达性分析在连续状态和动作空间中进行强化学习。虽然之前的工作主要集中在对抗攻击以实现鲁棒强化学习，但我们利用整组受干扰的输入来训练神经网络，并最大化最坏情况下的回报。可以验证，获得的代理比通过相关工作获得的代理更稳健，使它们更适用于安全关键的环境。对四个不同基准的广泛实证评估证明了这一点。



## **47. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

BaThe：通过将有害指令视为后门触发来防御多模式大型语言模型中的越狱攻击 cs.CR

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09093v1) [paper-pdf](http://arxiv.org/pdf/2408.09093v1)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.

摘要: 多通道大型语言模型(MLLM)在各种多通道任务中表现出令人印象深刻的性能。另一方面，附加图像模式的集成可能允许恶意用户在图像中注入有害内容以越狱。与基于文本的LLMS不同，在LLMS中，攻击者需要使用特定的算法选择离散的令牌来隐藏其恶意意图，而图像信号的连续性为攻击者提供了直接注入有害意图的机会。在这项工作中，我们提出了一种简单而有效的越狱防御机制--$\extbf{bathe}$($\extbf{ba}$ck door$\extbf{T}$rigger S$\extbf{h}$i$\extbf{e}$ld)。我们的工作是基于生成式语言模型对越狱后门攻击和虚拟提示后门攻击的最新研究。越狱后门攻击使用有害指令和手动创建的字符串作为触发器，使后门模型生成被禁止的响应。我们假设有害指令可以作为触发器，如果我们将拒绝响应设置为触发响应，那么反向模型就可以防御越狱攻击。我们通过利用虚拟拒绝提示来实现这一点，类似于虚拟提示后门攻击。我们将虚拟拒绝提示嵌入到软文本嵌入中，我们称之为‘’楔形‘’。我们的综合实验表明，BAIT有效地缓解了各种类型的越狱攻击，并且能够自适应地防御看不见的攻击，对MLLMS的性能影响最小。



## **48. HookChain: A new perspective for Bypassing EDR Solutions**

HookChain：询问EDR解决方案的新视角 cs.CR

50 pages, 23 figures, HookChain, Bypass EDR, Evading EDR, IAT Hook,  Halo's Gate

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2404.16856v3) [paper-pdf](http://arxiv.org/pdf/2404.16856v3)

**Authors**: Helvio Carvalho Junior

**Abstract**: In the current digital security ecosystem, where threats evolve rapidly and with complexity, companies developing Endpoint Detection and Response (EDR) solutions are in constant search for innovations that not only keep up but also anticipate emerging attack vectors. In this context, this article introduces the HookChain, a look from another perspective at widely known techniques, which when combined, provide an additional layer of sophisticated evasion against traditional EDR systems. Through a precise combination of IAT Hooking techniques, dynamic SSN resolution, and indirect system calls, HookChain redirects the execution flow of Windows subsystems in a way that remains invisible to the vigilant eyes of EDRs that only act on Ntdll.dll, without requiring changes to the source code of the applications and malwares involved. This work not only challenges current conventions in cybersecurity but also sheds light on a promising path for future protection strategies, leveraging the understanding that continuous evolution is key to the effectiveness of digital security. By developing and exploring the HookChain technique, this study significantly contributes to the body of knowledge in endpoint security, stimulating the development of more robust and adaptive solutions that can effectively address the ever-changing dynamics of digital threats. This work aspires to inspire deep reflection and advancement in the research and development of security technologies that are always several steps ahead of adversaries.

摘要: 在当前的数字安全生态系统中，威胁发展迅速且复杂，开发终端检测和响应(EDR)解决方案的公司正在不断寻找创新，不仅要跟上形势，还要预测新出现的攻击媒介。在此背景下，本文介绍了HookChain，从另一个角度介绍了广为人知的技术，这些技术结合在一起时，提供了针对传统EDR系统的另一层复杂规避。通过IAT挂钩技术、动态SSN解析和间接系统调用的精确组合，HookChain以一种仅作用于Ntdll.dll的EDR保持警惕的眼睛看不到的方式重定向Windows子系统的执行流，而不需要更改所涉及的应用程序和恶意软件的源代码。这项工作不仅挑战了目前的网络安全惯例，而且还揭示了未来保护战略的一条有希望的道路，充分利用了对持续演变是数字安全有效性的关键的理解。通过开发和探索HookChain技术，这项研究对终端安全方面的知识体系做出了重大贡献，刺激了能够有效应对不断变化的数字威胁动态的更健壮和适应性更强的解决方案的开发。这项工作旨在激发人们对安全技术研究和开发的深刻反思和进步，这些技术总是领先于对手几步。



## **49. Ask, Attend, Attack: A Effective Decision-Based Black-Box Targeted Attack for Image-to-Text Models**

询问、参与、攻击：针对图像到文本模型的有效基于决策的黑匣子定向攻击 cs.AI

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08989v1) [paper-pdf](http://arxiv.org/pdf/2408.08989v1)

**Authors**: Qingyuan Zeng, Zhenzhong Wang, Yiu-ming Cheung, Min Jiang

**Abstract**: While image-to-text models have demonstrated significant advancements in various vision-language tasks, they remain susceptible to adversarial attacks. Existing white-box attacks on image-to-text models require access to the architecture, gradients, and parameters of the target model, resulting in low practicality. Although the recently proposed gray-box attacks have improved practicality, they suffer from semantic loss during the training process, which limits their targeted attack performance. To advance adversarial attacks of image-to-text models, this paper focuses on a challenging scenario: decision-based black-box targeted attacks where the attackers only have access to the final output text and aim to perform targeted attacks. Specifically, we formulate the decision-based black-box targeted attack as a large-scale optimization problem. To efficiently solve the optimization problem, a three-stage process \textit{Ask, Attend, Attack}, called \textit{AAA}, is proposed to coordinate with the solver. \textit{Ask} guides attackers to create target texts that satisfy the specific semantics. \textit{Attend} identifies the crucial regions of the image for attacking, thus reducing the search space for the subsequent \textit{Attack}. \textit{Attack} uses an evolutionary algorithm to attack the crucial regions, where the attacks are semantically related to the target texts of \textit{Ask}, thus achieving targeted attacks without semantic loss. Experimental results on transformer-based and CNN+RNN-based image-to-text models confirmed the effectiveness of our proposed \textit{AAA}.

摘要: 虽然图像到文本模型在各种视觉语言任务中显示出了显著的进步，但它们仍然容易受到对手的攻击。现有的针对图像到文本模型的白盒攻击需要访问目标模型的体系结构、渐变和参数，导致实用性较低。最近提出的灰盒攻击虽然提高了实用性，但它们在训练过程中存在语义丢失问题，限制了它们的针对性攻击性能。为了推进图像到文本模型的对抗性攻击，本文重点研究了一个具有挑战性的场景：基于决策的黑箱定向攻击，攻击者只能访问最终的输出文本，并且目标是执行定向攻击。具体地说，我们将基于决策的黑盒定向攻击问题描述为一个大规模优化问题。为了有效地解决优化问题，提出了一个三阶段过程\textit{Ask}引导攻击者创建满足特定语义的目标文本。\textit{attend}识别图像中要攻击的关键区域，从而减少了后续\textit{攻击}的搜索空间。利用进化算法攻击与目标文本语义相关的关键区域，从而在不丢失语义的情况下实现目标攻击。在基于变压器和基于CNN+RNN的图文转换模型上的实验结果证实了该方法的有效性。



## **50. Stochastic Bandits Robust to Adversarial Attacks**

对对抗攻击具有鲁棒性的随机盗贼 cs.LG

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08859v1) [paper-pdf](http://arxiv.org/pdf/2408.08859v1)

**Authors**: Xuchuang Wang, Jinhang Zuo, Xutong Liu, John C. S. Lui, Mohammad Hajiesmaili

**Abstract**: This paper investigates stochastic multi-armed bandit algorithms that are robust to adversarial attacks, where an attacker can first observe the learner's action and {then} alter their reward observation. We study two cases of this model, with or without the knowledge of an attack budget $C$, defined as an upper bound of the summation of the difference between the actual and altered rewards. For both cases, we devise two types of algorithms with regret bounds having additive or multiplicative $C$ dependence terms. For the known attack budget case, we prove our algorithms achieve the regret bound of ${O}((K/\Delta)\log T + KC)$ and $\tilde{O}(\sqrt{KTC})$ for the additive and multiplicative $C$ terms, respectively, where $K$ is the number of arms, $T$ is the time horizon, $\Delta$ is the gap between the expected rewards of the optimal arm and the second-best arm, and $\tilde{O}$ hides the logarithmic factors. For the unknown case, we prove our algorithms achieve the regret bound of $\tilde{O}(\sqrt{KT} + KC^2)$ and $\tilde{O}(KC\sqrt{T})$ for the additive and multiplicative $C$ terms, respectively. In addition to these upper bound results, we provide several lower bounds showing the tightness of our bounds and the optimality of our algorithms. These results delineate an intrinsic separation between the bandits with attacks and corruption models [Lykouris et al., 2018].

摘要: 研究了对敌方攻击具有鲁棒性的随机多臂盗贼算法，其中攻击者可以首先观察到学习者的行为，然后改变他们的奖励观察。我们研究了该模型的两种情况，在有或不知道攻击预算$C$的情况下，该预算被定义为实际和改变的奖励之间的差值之和的上界。对于这两种情况，我们设计了两种类型的算法，它们具有加性或乘性$C$依赖项的后悔界。对于已知的攻击预算情形，我们证明了我们的算法对于加性和乘性$C$项分别达到了遗憾界$O}((K/\Delta)\log T+KC)$和$\tide{O}(\Sqrt{KTC})$，其中$K$是武器数，$T$是时间范围，$\Delta$是最优ARM和次优ARM的期望回报之间的差距，$\tide{O}$隐藏了对数因子。对于未知情况，我们证明了对于加性项和乘性项，我们的算法分别达到了[0}(KT}+KC^2)$和[O](KCSQRT{T})$的遗憾界。除了这些上界结果外，我们还提供了几个下界，表明了我们的界的紧密性和我们的算法的最优性。这些结果描绘了具有攻击和腐败模式的土匪之间的内在分离[Lykouris等人，2018年]。



