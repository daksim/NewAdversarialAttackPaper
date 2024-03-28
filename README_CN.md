# Latest Adversarial Attack Papers
**update at 2024-03-28 11:36:33**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Deep Learning for Robust and Explainable Models in Computer Vision**

用于计算机视觉中鲁棒和可解释模型的深度学习 cs.CV

150 pages, 37 figures, 12 tables

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18674v1) [paper-pdf](http://arxiv.org/pdf/2403.18674v1)

**Authors**: Mohammadreza Amirian

**Abstract**: Recent breakthroughs in machine and deep learning (ML and DL) research have provided excellent tools for leveraging enormous amounts of data and optimizing huge models with millions of parameters to obtain accurate networks for image processing. These developments open up tremendous opportunities for using artificial intelligence (AI) in the automation and human assisted AI industry. However, as more and more models are deployed and used in practice, many challenges have emerged. This thesis presents various approaches that address robustness and explainability challenges for using ML and DL in practice.   Robustness and reliability are the critical components of any model before certification and deployment in practice. Deep convolutional neural networks (CNNs) exhibit vulnerability to transformations of their inputs, such as rotation and scaling, or intentional manipulations as described in the adversarial attack literature. In addition, building trust in AI-based models requires a better understanding of current models and developing methods that are more explainable and interpretable a priori.   This thesis presents developments in computer vision models' robustness and explainability. Furthermore, this thesis offers an example of using vision models' feature response visualization (models' interpretations) to improve robustness despite interpretability and robustness being seemingly unrelated in the related research. Besides methodological developments for robust and explainable vision models, a key message of this thesis is introducing model interpretation techniques as a tool for understanding vision models and improving their design and robustness. In addition to the theoretical developments, this thesis demonstrates several applications of ML and DL in different contexts, such as medical imaging and affective computing.

摘要: 机器和深度学习(ML和DL)研究的最新突破为利用海量数据和优化具有数百万参数的巨大模型提供了极好的工具，以获得用于图像处理的准确网络。这些发展为人工智能(AI)在自动化和人工辅助AI行业中的使用打开了巨大的机会。然而，随着越来越多的模型在实践中部署和使用，出现了许多挑战。这篇论文提出了各种方法来解决在实践中使用ML和DL时的健壮性和可解释性挑战。在实践中认证和部署之前，健壮性和可靠性是任何模型的关键组件。深层卷积神经网络(CNN)表现出对其输入的变换的脆弱性，例如旋转和缩放，或者如对抗性攻击文献中所描述的故意操纵。此外，建立对基于人工智能的模型的信任需要更好地理解当前的模型，并开发更具解释性和先验性的方法。本文介绍了计算机视觉模型的稳健性和可解释性方面的研究进展。此外，本文还给出了一个使用视觉模型的特征响应可视化(模型的解释)来提高稳健性的例子，尽管可解释性和稳健性在相关研究中似乎是无关的。除了稳健和可解释的视觉模型的方法论发展外，本文的一个关键信息是引入模型解释技术作为理解视觉模型的工具，并改进其设计和稳健性。除了理论上的发展，本文还展示了ML和DL在不同环境中的几个应用，例如医学成像和情感计算。



## **2. LCANets++: Robust Audio Classification using Multi-layer Neural Networks with Lateral Competition**

LCANets ++：使用具有横向竞争的多层神经网络的鲁棒音频分类 cs.SD

Accepted at 2024 IEEE International Conference on Acoustics, Speech  and Signal Processing Workshops (ICASSPW)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2308.12882v2) [paper-pdf](http://arxiv.org/pdf/2308.12882v2)

**Authors**: Sayanton V. Dibbo, Juston S. Moore, Garrett T. Kenyon, Michael A. Teti

**Abstract**: Audio classification aims at recognizing audio signals, including speech commands or sound events. However, current audio classifiers are susceptible to perturbations and adversarial attacks. In addition, real-world audio classification tasks often suffer from limited labeled data. To help bridge these gaps, previous work developed neuro-inspired convolutional neural networks (CNNs) with sparse coding via the Locally Competitive Algorithm (LCA) in the first layer (i.e., LCANets) for computer vision. LCANets learn in a combination of supervised and unsupervised learning, reducing dependency on labeled samples. Motivated by the fact that auditory cortex is also sparse, we extend LCANets to audio recognition tasks and introduce LCANets++, which are CNNs that perform sparse coding in multiple layers via LCA. We demonstrate that LCANets++ are more robust than standard CNNs and LCANets against perturbations, e.g., background noise, as well as black-box and white-box attacks, e.g., evasion and fast gradient sign (FGSM) attacks.

摘要: 音频分类的目的是识别音频信号，包括语音命令或声音事件。然而，当前的音频分类器容易受到扰动和对抗性攻击。此外，现实世界的音频分类任务通常会受到有限的标签数据的影响。为了弥补这些差距，以前的工作发展了神经启发卷积神经网络(CNN)，通过第一层的局部竞争算法(LCA)进行稀疏编码，用于计算机视觉。LCANet在监督和非监督学习的组合中学习，减少了对标记样本的依赖。基于听觉皮层也是稀疏的这一事实，我们将LCANets扩展到音频识别任务，并引入LCANets++，LCANets++是通过LCA在多层进行稀疏编码的CNN。我们证明了LCANet++比标准的CNN和LCANet对扰动(例如背景噪声)以及黑盒和白盒攻击(例如逃避和快速梯度符号(FGSM)攻击)具有更强的鲁棒性。



## **3. The Impact of Uniform Inputs on Activation Sparsity and Energy-Latency Attacks in Computer Vision**

均匀输入对计算机视觉中激活稀疏性和能量延迟攻击的影响 cs.CR

Accepted at the DLSP 2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18587v1) [paper-pdf](http://arxiv.org/pdf/2403.18587v1)

**Authors**: Andreas Müller, Erwin Quiring

**Abstract**: Resource efficiency plays an important role for machine learning nowadays. The energy and decision latency are two critical aspects to ensure a sustainable and practical application. Unfortunately, the energy consumption and decision latency are not robust against adversaries. Researchers have recently demonstrated that attackers can compute and submit so-called sponge examples at inference time to increase the energy consumption and decision latency of neural networks. In computer vision, the proposed strategy crafts inputs with less activation sparsity which could otherwise be used to accelerate the computation. In this paper, we analyze the mechanism how these energy-latency attacks reduce activation sparsity. In particular, we find that input uniformity is a key enabler. A uniform image, that is, an image with mostly flat, uniformly colored surfaces, triggers more activations due to a specific interplay of convolution, batch normalization, and ReLU activation. Based on these insights, we propose two new simple, yet effective strategies for crafting sponge examples: sampling images from a probability distribution and identifying dense, yet inconspicuous inputs in natural datasets. We empirically examine our findings in a comprehensive evaluation with multiple image classification models and show that our attack achieves the same sparsity effect as prior sponge-example methods, but at a fraction of computation effort. We also show that our sponge examples transfer between different neural networks. Finally, we discuss applications of our findings for the good by improving efficiency by increasing sparsity.

摘要: 资源效率在当今机器学习中扮演着重要的角色。能量和决策延迟是确保可持续和实际应用的两个关键方面。不幸的是，能量消耗和决策延迟对对手的健壮性不强。研究人员最近证明，攻击者可以在推理时计算并提交所谓的海绵示例，以增加神经网络的能量消耗和决策延迟。在计算机视觉中，所提出的策略以较小的激活稀疏性来制作输入，否则可以用来加速计算。在本文中，我们分析了这些能量延迟攻击降低激活稀疏性的机制。特别是，我们发现输入的一致性是一个关键的推动因素。由于卷积、批处理归一化和REU激活的特定相互作用，统一图像，即具有大部分平坦、统一颜色的表面的图像，会触发更多的激活。基于这些见解，我们提出了两种新的简单而有效的海绵样本制作策略：从概率分布中采样图像，以及在自然数据集中识别密集但不明显的输入。我们在多个图像分类模型的综合评估中对我们的发现进行了实证检验，结果表明，我们的攻击达到了与以前的海绵示例方法相同的稀疏效果，但计算量只有很小一部分。我们还表明，我们的海绵样本在不同的神经网络之间转换。最后，我们讨论了我们的发现的应用，通过增加稀疏性来提高效率。



## **4. MisGUIDE : Defense Against Data-Free Deep Learning Model Extraction**

MisGUIDE：防御无数据深度学习模型提取 cs.CR

Under Review

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18580v1) [paper-pdf](http://arxiv.org/pdf/2403.18580v1)

**Authors**: Mahendra Gurve, Sankar Behera, Satyadev Ahlawat, Yamuna Prasad

**Abstract**: The rise of Machine Learning as a Service (MLaaS) has led to the widespread deployment of machine learning models trained on diverse datasets. These models are employed for predictive services through APIs, raising concerns about the security and confidentiality of the models due to emerging vulnerabilities in prediction APIs. Of particular concern are model cloning attacks, where individuals with limited data and no knowledge of the training dataset manage to replicate a victim model's functionality through black-box query access. This commonly entails generating adversarial queries to query the victim model, thereby creating a labeled dataset.   This paper proposes "MisGUIDE", a two-step defense framework for Deep Learning models that disrupts the adversarial sample generation process by providing a probabilistic response when the query is deemed OOD. The first step employs a Vision Transformer-based framework to identify OOD queries, while the second step perturbs the response for such queries, introducing a probabilistic loss function to MisGUIDE the attackers. The aim of the proposed defense method is to reduce the accuracy of the cloned model while maintaining accuracy on authentic queries. Extensive experiments conducted on two benchmark datasets demonstrate that the proposed framework significantly enhances the resistance against state-of-the-art data-free model extraction in black-box settings.

摘要: 机器学习即服务(MLaaS)的兴起导致了在不同数据集上训练的机器学习模型的广泛部署。这些模型通过API用于预测服务，由于预测API中新出现的漏洞，人们对模型的安全性和保密性表示担忧。特别令人关切的是模型克隆攻击，数据有限且不了解训练数据集的个人设法通过黑盒查询访问复制受害者模型的功能。这通常需要生成敌意查询以查询受害者模型，从而创建标签数据集。本文提出了一种用于深度学习模型的两步防御框架“MisGuide”，该框架通过在查询被认为是面向对象时提供概率响应来中断敌意样本的生成过程。第一步使用基于Vision Transformer的框架来识别OOD查询，而第二步干扰对此类查询的响应，引入概率损失函数来误导攻击者。该防御方法的目的是降低克隆模型的准确性，同时保持对真实查询的准确性。在两个基准数据集上进行的广泛实验表明，该框架显著增强了对黑盒环境下最先进的无数据模型提取的抵抗力。



## **5. CosalPure: Learning Concept from Group Images for Robust Co-Saliency Detection**

CosalPure：基于组图像的学习概念，用于鲁棒共显著性检测 cs.CV

8 pages

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18554v1) [paper-pdf](http://arxiv.org/pdf/2403.18554v1)

**Authors**: Jiayi Zhu, Qing Guo, Felix Juefei-Xu, Yihao Huang, Yang Liu, Geguang Pu

**Abstract**: Co-salient object detection (CoSOD) aims to identify the common and salient (usually in the foreground) regions across a given group of images. Although achieving significant progress, state-of-the-art CoSODs could be easily affected by some adversarial perturbations, leading to substantial accuracy reduction. The adversarial perturbations can mislead CoSODs but do not change the high-level semantic information (e.g., concept) of the co-salient objects. In this paper, we propose a novel robustness enhancement framework by first learning the concept of the co-salient objects based on the input group images and then leveraging this concept to purify adversarial perturbations, which are subsequently fed to CoSODs for robustness enhancement. Specifically, we propose CosalPure containing two modules, i.e., group-image concept learning and concept-guided diffusion purification. For the first module, we adopt a pre-trained text-to-image diffusion model to learn the concept of co-salient objects within group images where the learned concept is robust to adversarial examples. For the second module, we map the adversarial image to the latent space and then perform diffusion generation by embedding the learned concept into the noise prediction function as an extra condition. Our method can effectively alleviate the influence of the SOTA adversarial attack containing different adversarial patterns, including exposure and noise. The extensive results demonstrate that our method could enhance the robustness of CoSODs significantly.

摘要: 共显著目标检测(CoSOD)的目的是识别给定图像组中的共同和显著(通常在前景中)区域。虽然取得了重大进展，但最先进的CoSOD很容易受到一些对抗性扰动的影响，导致精度大幅下降。对抗性扰动会误导CoSOD，但不会改变共显著对象的高级语义信息(例如，概念)。在本文中，我们提出了一种新的稳健性增强框架，该框架首先学习基于输入分组图像的共显著对象的概念，然后利用该概念来净化对抗性扰动，然后将这些扰动馈送到CoSOD以增强稳健性。具体地说，我们提出CosalPure包含两个模块，即组图像概念学习和概念引导的扩散净化。对于第一个模块，我们采用预先训练的文本到图像的扩散模型来学习组图像中共显著对象的概念，其中学习的概念对对抗性例子是健壮的。对于第二个模块，我们将敌意图像映射到潜在空间，然后通过将学习到的概念作为附加条件嵌入到噪声预测函数中来执行扩散生成。我们的方法可以有效地缓解SOTA对抗性攻击的影响，该攻击包含不同的对抗性模式，包括暴露和噪声。实验结果表明，该方法可以显著提高CoSOD的稳健性。



## **6. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

披着羊皮的狼：广义嵌套越狱陷阱可以轻松愚弄大型语言模型 cs.CL

Acccepted by NAACL 2024, 18 pages, 7 figures, 13 tables

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2311.08268v3) [paper-pdf](http://arxiv.org/pdf/2311.08268v3)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, which compromises either generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.

摘要: 大型语言模型(LLM)，如ChatGPT和GPT-4，旨在提供有用和安全的响应。然而，被称为“越狱”的对抗性提示可能会绕过安全措施，导致LLMS生成潜在的有害内容。探索越狱提示可以帮助更好地揭示LLM的弱点，并进一步指导我们确保它们的安全。不幸的是，现有的越狱方法要么需要复杂的人工设计，要么需要对其他白盒模型进行优化，这要么损害了通用性，要么影响了效率。本文将越狱提示攻击概括为两个方面：(1)提示重写和(2)场景嵌套。在此基础上，我们提出了ReNeLLM，这是一个利用LLM自身生成有效越狱提示的自动化框架。广泛的实验表明，与现有的基准相比，ReNeLLM显著提高了攻击成功率，同时大大降低了时间成本。我们的研究也揭示了现有防御方法在保护低密度脂蛋白方面的不足。最后，从即时执行优先级的角度分析了LLMS防御失败的原因，并提出了相应的防御策略。我们希望我们的研究能够促进学术界和低成本管理系统开发商提供更安全和更规范的低成本管理系统。代码可在https://github.com/NJUNLP/ReNeLLM.上获得



## **7. Attack and Defense Analysis of Learned Image Compression**

学习图像压缩的攻防分析 eess.IV

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2401.10345v3) [paper-pdf](http://arxiv.org/pdf/2401.10345v3)

**Authors**: Tianyu Zhu, Heming Sun, Xiankui Xiong, Xuanpeng Zhu, Yong Gong, Minge jing, Yibo Fan

**Abstract**: Learned image compression (LIC) is becoming more and more popular these years with its high efficiency and outstanding compression quality. Still, the practicality against modified inputs added with specific noise could not be ignored. White-box attacks such as FGSM and PGD use only gradient to compute adversarial images that mislead LIC models to output unexpected results. Our experiments compare the effects of different dimensions such as attack methods, models, qualities, and targets, concluding that in the worst case, there is a 61.55% decrease in PSNR or a 19.15 times increase in bpp under the PGD attack. To improve their robustness, we conduct adversarial training by adding adversarial images into the training datasets, which obtains a 95.52% decrease in the R-D cost of the most vulnerable LIC model. We further test the robustness of H.266, whose better performance on reconstruction quality extends its possibility to defend one-step or iterative adversarial attacks.

摘要: 近年来，学习图像压缩（LIC）以其高效率和优异的压缩质量得到了越来越广泛的应用。然而，对于添加特定噪声的修改输入的实用性也不容忽视。FGSM和PGD等白盒攻击仅使用梯度来计算对抗图像，从而误导LIC模型以输出意外结果。我们的实验比较了攻击方法、模型、质量和目标等不同维度的影响，得出的结论是，在最坏的情况下，PGD攻击下，PSNR下降了61.55%，bpp增加了19.15倍。为了提高它们的鲁棒性，我们通过将对抗图像添加到训练数据集中来进行对抗训练，这使得最脆弱的LIC模型的R—D成本降低了95.52%。我们进一步测试了H.266的鲁棒性，其在重建质量方面的更好性能扩展了其防御一步或迭代对抗攻击的可能性。



## **8. $\textit{LinkPrompt}$: Natural and Universal Adversarial Attacks on Prompt-based Language Models**

$\textit {LinkPrompt}$：基于XSLT语言模型的自然和普遍对抗攻击 cs.CL

Accepted to the main conference of NAACL2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.16432v2) [paper-pdf](http://arxiv.org/pdf/2403.16432v2)

**Authors**: Yue Xu, Wenjie Wang

**Abstract**: Prompt-based learning is a new language model training paradigm that adapts the Pre-trained Language Models (PLMs) to downstream tasks, which revitalizes the performance benchmarks across various natural language processing (NLP) tasks. Instead of using a fixed prompt template to fine-tune the model, some research demonstrates the effectiveness of searching for the prompt via optimization. Such prompt optimization process of prompt-based learning on PLMs also gives insight into generating adversarial prompts to mislead the model, raising concerns about the adversarial vulnerability of this paradigm. Recent studies have shown that universal adversarial triggers (UATs) can be generated to alter not only the predictions of the target PLMs but also the prediction of corresponding Prompt-based Fine-tuning Models (PFMs) under the prompt-based learning paradigm. However, UATs found in previous works are often unreadable tokens or characters and can be easily distinguished from natural texts with adaptive defenses. In this work, we consider the naturalness of the UATs and develop $\textit{LinkPrompt}$, an adversarial attack algorithm to generate UATs by a gradient-based beam search algorithm that not only effectively attacks the target PLMs and PFMs but also maintains the naturalness among the trigger tokens. Extensive results demonstrate the effectiveness of $\textit{LinkPrompt}$, as well as the transferability of UATs generated by $\textit{LinkPrompt}$ to open-sourced Large Language Model (LLM) Llama2 and API-accessed LLM GPT-3.5-turbo.

摘要: 基于提示的学习是一种新的语言模型训练范式，它使预先训练的语言模型(PLM)适应于下游任务，从而重振各种自然语言处理(NLP)任务的表现基准。一些研究证明了通过优化来搜索提示的有效性，而不是使用固定的提示模板来微调模型。这种基于提示的PLM学习的快速优化过程也为生成对抗性提示以误导模型提供了洞察力，这引发了人们对这种范式的对抗性脆弱性的担忧。最近的研究表明，在基于提示的学习范式下，通用对抗触发器(UAT)不仅可以改变目标PLM的预测，还可以改变相应的基于提示的精调模型(PFM)的预测。然而，在以前的著作中发现的UAT通常是不可读的符号或字符，并且可以很容易地与具有自适应防御的自然文本区分开来。在这项工作中，我们考虑了UAT的自然性，并开发了一种对抗性攻击算法，通过基于梯度的波束搜索算法来生成UAT，该算法不仅有效地攻击了目标PLM和PPM，而且保持了触发令牌之间的自然度。广泛的结果证明了$\textit{LinkPrompt}$的有效性，以及由$\textit{LinkPrompt}$生成的UAT可以移植到开源的大型语言模型(LLM)Llama2和API访问的LLm GPT-3.5-Turbo。



## **9. SemRoDe: Macro Adversarial Training to Learn Representations That are Robust to Word-Level Attacks**

SemRoDe：宏对抗训练，学习对单词级攻击具有鲁棒性的表示 cs.CL

Published in NAACL 2024 (Main Track)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18423v1) [paper-pdf](http://arxiv.org/pdf/2403.18423v1)

**Authors**: Brian Formento, Wenjie Feng, Chuan Sheng Foo, Luu Anh Tuan, See-Kiong Ng

**Abstract**: Language models (LMs) are indispensable tools for natural language processing tasks, but their vulnerability to adversarial attacks remains a concern. While current research has explored adversarial training techniques, their improvements to defend against word-level attacks have been limited. In this work, we propose a novel approach called Semantic Robust Defence (SemRoDe), a Macro Adversarial Training strategy to enhance the robustness of LMs. Drawing inspiration from recent studies in the image domain, we investigate and later confirm that in a discrete data setting such as language, adversarial samples generated via word substitutions do indeed belong to an adversarial domain exhibiting a high Wasserstein distance from the base domain. Our method learns a robust representation that bridges these two domains. We hypothesize that if samples were not projected into an adversarial domain, but instead to a domain with minimal shift, it would improve attack robustness. We align the domains by incorporating a new distance-based objective. With this, our model is able to learn more generalized representations by aligning the model's high-level output features and therefore better handling unseen adversarial samples. This method can be generalized across word embeddings, even when they share minimal overlap at both vocabulary and word-substitution levels. To evaluate the effectiveness of our approach, we conduct experiments on BERT and RoBERTa models on three datasets. The results demonstrate promising state-of-the-art robustness.

摘要: 语言模型(LMS)是自然语言处理任务中不可或缺的工具，但其易受敌意攻击的问题仍是一个令人担忧的问题。虽然目前的研究已经探索了对抗性训练技术，但它们在防御单词级攻击方面的改进有限。在这项工作中，我们提出了一种新的方法，称为语义稳健防御(SemRoDe)，这是一种宏观对抗性训练策略，以增强LMS的健壮性。受图像领域最新研究的启发，我们调查并证实，在语言等离散数据环境中，通过词替换生成的对抗性样本确实属于与基域有较高Wasserstein距离的对抗性领域。我们的方法学习了连接这两个域的健壮表示。我们假设，如果样本不被投影到敌对域，而是投影到位移最小的域，将提高攻击的稳健性。我们通过结合一个新的基于距离的目标来对齐域。有了这一点，我们的模型能够通过对齐模型的高级输出特征来学习更通用的表示，从而更好地处理看不见的敌意样本。这种方法可以在单词嵌入中推广，即使它们在词汇和单词替换级别上共享最小的重叠。为了评估该方法的有效性，我们在三个数据集上对Bert和Roberta模型进行了实验。结果表明，该方法具有良好的稳健性。



## **10. LocalStyleFool: Regional Video Style Transfer Attack Using Segment Anything Model**

LocalStyleFool：基于段任意模型的区域视频风格转移攻击 cs.CV

Accepted to 2024 IEEE Security and Privacy Workshops (SPW)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.11656v2) [paper-pdf](http://arxiv.org/pdf/2403.11656v2)

**Authors**: Yuxin Cao, Jinghao Li, Xi Xiao, Derui Wang, Minhui Xue, Hao Ge, Wei Liu, Guangwu Hu

**Abstract**: Previous work has shown that well-crafted adversarial perturbations can threaten the security of video recognition systems. Attackers can invade such models with a low query budget when the perturbations are semantic-invariant, such as StyleFool. Despite the query efficiency, the naturalness of the minutia areas still requires amelioration, since StyleFool leverages style transfer to all pixels in each frame. To close the gap, we propose LocalStyleFool, an improved black-box video adversarial attack that superimposes regional style-transfer-based perturbations on videos. Benefiting from the popularity and scalably usability of Segment Anything Model (SAM), we first extract different regions according to semantic information and then track them through the video stream to maintain the temporal consistency. Then, we add style-transfer-based perturbations to several regions selected based on the associative criterion of transfer-based gradient information and regional area. Perturbation fine adjustment is followed to make stylized videos adversarial. We demonstrate that LocalStyleFool can improve both intra-frame and inter-frame naturalness through a human-assessed survey, while maintaining competitive fooling rate and query efficiency. Successful experiments on the high-resolution dataset also showcase that scrupulous segmentation of SAM helps to improve the scalability of adversarial attacks under high-resolution data.

摘要: 以往的工作表明，精心设计的对抗性扰动会威胁到视频识别系统的安全性。当扰动是语义不变的(如StyleFool)时，攻击者可以用较低的查询预算入侵这样的模型。尽管查询效率很高，但细节区域的自然度仍然需要改进，因为StyleFool利用样式传递到每帧中的所有像素。为了缩小这一差距，我们提出了LocalStyleFool，一种改进的黑盒视频对抗性攻击，将基于区域风格转移的扰动叠加到视频上。利用Segment Anything Model(SAM)的普及性和可伸缩性，我们首先根据语义信息提取不同的区域，然后通过视频流对它们进行跟踪，以保持时间一致性。然后，我们根据基于转移的梯度信息和区域面积的关联准则，将基于风格转移的扰动添加到选择的多个区域。随后进行微调，使风格化视频具有对抗性。通过一个人工评估的调查，我们证明了LocalStyleFool可以提高帧内和帧间的自然度，同时保持有竞争力的愚弄率和查询效率。在高分辨率数据集上的成功实验也表明，对SAM进行严格的分割有助于提高高分辨率数据下对抗性攻击的可扩展性。



## **11. Physical 3D Adversarial Attacks against Monocular Depth Estimation in Autonomous Driving**

自主驾驶中单目深度估计的物理3D对抗攻击 cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.17301v2) [paper-pdf](http://arxiv.org/pdf/2403.17301v2)

**Authors**: Junhao Zheng, Chenhao Lin, Jiahao Sun, Zhengyu Zhao, Qian Li, Chao Shen

**Abstract**: Deep learning-based monocular depth estimation (MDE), extensively applied in autonomous driving, is known to be vulnerable to adversarial attacks. Previous physical attacks against MDE models rely on 2D adversarial patches, so they only affect a small, localized region in the MDE map but fail under various viewpoints. To address these limitations, we propose 3D Depth Fool (3D$^2$Fool), the first 3D texture-based adversarial attack against MDE models. 3D$^2$Fool is specifically optimized to generate 3D adversarial textures agnostic to model types of vehicles and to have improved robustness in bad weather conditions, such as rain and fog. Experimental results validate the superior performance of our 3D$^2$Fool across various scenarios, including vehicles, MDE models, weather conditions, and viewpoints. Real-world experiments with printed 3D textures on physical vehicle models further demonstrate that our 3D$^2$Fool can cause an MDE error of over 10 meters.

摘要: 基于深度学习的单目深度估计（MDE）广泛应用于自动驾驶中，已知容易受到对抗性攻击。以前针对MDE模型的物理攻击依赖于2D对抗补丁，因此它们只影响MDE地图中的一个小的局部区域，但在不同的视角下都会失败。为了解决这些限制，我们提出了3D Depth Fool（3D$^2$Fool），这是第一个针对MDE模型的基于3D纹理的对抗攻击。3D$^2$Fool经过专门优化，以生成3D对抗纹理，这些纹理不可知，以模型类型的车辆，并在恶劣天气条件下（如雨和雾）具有更高的鲁棒性。实验结果验证了我们的3D$^2$Fool在各种场景中的卓越性能，包括车辆、MDE模型、天气条件和视点。在实体车辆模型上使用打印3D纹理的真实实验进一步表明，我们的3D$^2$Fool可以导致超过10米的MDE误差。



## **12. Uncertainty-Aware SAR ATR: Defending Against Adversarial Attacks via Bayesian Neural Networks**

不确定性感知SAR ATR：利用贝叶斯神经网络防御对抗攻击 cs.CV

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18318v1) [paper-pdf](http://arxiv.org/pdf/2403.18318v1)

**Authors**: Tian Ye, Rajgopal Kannan, Viktor Prasanna, Carl Busart

**Abstract**: Adversarial attacks have demonstrated the vulnerability of Machine Learning (ML) image classifiers in Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) systems. An adversarial attack can deceive the classifier into making incorrect predictions by perturbing the input SAR images, for example, with a few scatterers attached to the on-ground objects. Therefore, it is critical to develop robust SAR ATR systems that can detect potential adversarial attacks by leveraging the inherent uncertainty in ML classifiers, thereby effectively alerting human decision-makers. In this paper, we propose a novel uncertainty-aware SAR ATR for detecting adversarial attacks. Specifically, we leverage the capability of Bayesian Neural Networks (BNNs) in performing image classification with quantified epistemic uncertainty to measure the confidence for each input SAR image. By evaluating the uncertainty, our method alerts when the input SAR image is likely to be adversarially generated. Simultaneously, we also generate visual explanations that reveal the specific regions in the SAR image where the adversarial scatterers are likely to to be present, thus aiding human decision-making with hints of evidence of adversarial attacks. Experiments on the MSTAR dataset demonstrate that our approach can identify over 80% adversarial SAR images with fewer than 20% false alarms, and our visual explanations can identify up to over 90% of scatterers in an adversarial SAR image.

摘要: 敌意攻击已经证明了合成孔径雷达(SAR)自动目标识别(ATR)系统中机器学习(ML)图像分类器的脆弱性。对抗性攻击可以通过干扰输入的SAR图像来欺骗分类器做出错误的预测，例如，将一些散射体连接到地面对象上。因此，开发稳健的SAR ATR系统至关重要，它可以利用ML分类器中固有的不确定性来检测潜在的对手攻击，从而有效地向人类决策者发出警报。在本文中，我们提出了一种新的不确定性感知的SARATR来检测敌意攻击。具体地说，我们利用贝叶斯神经网络(BNN)在量化认知不确定性的情况下执行图像分类的能力来衡量每一幅输入SAR图像的置信度。通过评估不确定性，当输入的SAR图像可能被恶意生成时，我们的方法会发出警报。同时，我们还生成视觉解释，揭示SAR图像中可能存在对抗性散射体的特定区域，从而为人类决策提供对抗性攻击的证据提示。在MStar数据集上的实验表明，我们的方法可以识别80%以上的对抗性SAR图像，虚警率不到20%，我们的视觉解释可以识别高达90%以上的对抗性SAR图像中的散射体。



## **13. Bayesian Learned Models Can Detect Adversarial Malware For Free**

贝叶斯学习模型可以免费检测对抗性恶意软件 cs.CR

Accepted to the 29th European Symposium on Research in Computer  Security (ESORICS) 2024 Conference

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18309v1) [paper-pdf](http://arxiv.org/pdf/2403.18309v1)

**Authors**: Bao Gia Doan, Dang Quang Nguyen, Paul Montague, Tamas Abraham, Olivier De Vel, Seyit Camtepe, Salil S. Kanhere, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: The vulnerability of machine learning-based malware detectors to adversarial attacks has prompted the need for robust solutions. Adversarial training is an effective method but is computationally expensive to scale up to large datasets and comes at the cost of sacrificing model performance for robustness. We hypothesize that adversarial malware exploits the low-confidence regions of models and can be identified using epistemic uncertainty of ML approaches -- epistemic uncertainty in a machine learning-based malware detector is a result of a lack of similar training samples in regions of the problem space. In particular, a Bayesian formulation can capture the model parameters' distribution and quantify epistemic uncertainty without sacrificing model performance. To verify our hypothesis, we consider Bayesian learning approaches with a mutual information-based formulation to quantify uncertainty and detect adversarial malware in Android, Windows domains and PDF malware. We found, quantifying uncertainty through Bayesian learning methods can defend against adversarial malware. In particular, Bayesian models: (1) are generally capable of identifying adversarial malware in both feature and problem space, (2) can detect concept drift by measuring uncertainty, and (3) with a diversity-promoting approach (or better posterior approximations) lead to parameter instances from the posterior to significantly enhance a detectors' ability.

摘要: 基于机器学习的恶意软件检测器对对手攻击的脆弱性促使人们需要强大的解决方案。对抗性训练是一种有效的方法，但扩大到大型数据集的计算成本很高，而且是以牺牲模型性能来换取健壮性为代价的。我们假设敌意恶意软件利用了模型的低置信度区域，并可以使用ML方法的认知不确定性进行识别--基于机器学习的恶意软件检测器中的认知不确定性是由于问题空间区域中缺乏类似的训练样本造成的。特别是，贝叶斯公式可以捕捉模型参数的分布，并在不牺牲模型性能的情况下量化认知不确定性。为了验证我们的假设，我们考虑了基于互信息的贝叶斯学习方法来量化不确定性，并检测Android、Windows域和PDF恶意软件中的恶意软件。我们发现，通过贝叶斯学习方法量化不确定性可以防御敌意恶意软件。特别是，贝叶斯模型：(1)通常能够在特征和问题空间中识别恶意软件；(2)可以通过测量不确定性来检测概念漂移；(3)通过促进多样性的方法(或更好的后验近似)从后验获得参数实例，从而显著增强检测器的能力。



## **14. Bidirectional Consistency Models**

双向一致性模型 cs.LG

40 pages, 25 figures

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.18035v1) [paper-pdf](http://arxiv.org/pdf/2403.18035v1)

**Authors**: Liangchen Li, Jiajun He

**Abstract**: Diffusion models (DMs) are capable of generating remarkably high-quality samples by iteratively denoising a random vector, a process that corresponds to moving along the probability flow ordinary differential equation (PF ODE). Interestingly, DMs can also invert an input image to noise by moving backward along the PF ODE, a key operation for downstream tasks such as interpolation and image editing. However, the iterative nature of this process restricts its speed, hindering its broader application. Recently, Consistency Models (CMs) have emerged to address this challenge by approximating the integral of the PF ODE, thereby bypassing the need to iterate. Yet, the absence of an explicit ODE solver complicates the inversion process. To resolve this, we introduce the Bidirectional Consistency Model (BCM), which learns a single neural network that enables both forward and backward traversal along the PF ODE, efficiently unifying generation and inversion tasks within one framework. Notably, our proposed method enables one-step generation and inversion while also allowing the use of additional steps to enhance generation quality or reduce reconstruction error. Furthermore, by leveraging our model's bidirectional consistency, we introduce a sampling strategy that can enhance FID while preserving the generated image content. We further showcase our model's capabilities in several downstream tasks, such as interpolation and inpainting, and present demonstrations of potential applications, including blind restoration of compressed images and defending black-box adversarial attacks.

摘要: 扩散模型(DM)能够通过迭代地对随机向量去噪来生成非常高质量的样本，该过程对应于沿着概率流常微分方程式(PF ODE)移动。有趣的是，DM还可以通过沿PF ODE向后移动来将输入图像反转为噪声，这是下游任务(如插补和图像编辑)的关键操作。然而，这一过程的迭代性质限制了其速度，阻碍了其更广泛的应用。最近，一致性模型(CM)已经出现，通过近似PF ODE的积分来解决这一挑战，从而绕过了迭代的需要。然而，由于没有显式的常微分方程组解算器，使得反演过程变得更加复杂。为了解决这个问题，我们引入了双向一致性模型(BCM)，它学习一个单一的神经网络，允许沿着PF ODE进行前向和后向遍历，有效地将生成和反转任务统一在一个框架内。值得注意的是，我们提出的方法支持一步生成和反转，同时还允许使用额外的步骤来提高生成质量或减少重建误差。此外，通过利用模型的双向一致性，我们引入了一种采样策略，该策略可以在保留生成的图像内容的同时增强FID。我们进一步展示了我们的模型在几个下游任务中的能力，如插补和修复，并展示了潜在的应用，包括压缩图像的盲恢复和防御黑盒攻击。



## **15. Analyzing the Quality Attributes of AI Vision Models in Open Repositories Under Adversarial Attacks**

对抗攻击下开放仓库中人工智能视觉模型的质量属性分析 cs.CR

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2401.12261v2) [paper-pdf](http://arxiv.org/pdf/2401.12261v2)

**Authors**: Zerui Wang, Yan Liu

**Abstract**: As AI models rapidly evolve, they are frequently released to open repositories, such as HuggingFace. It is essential to perform quality assurance validation on these models before integrating them into the production development lifecycle. In addition to evaluating efficiency in terms of balanced accuracy and computing costs, adversarial attacks are potential threats to the robustness and explainability of AI models. Meanwhile, XAI applies algorithms that approximate inputs to outputs post-hoc to identify the contributing features. Adversarial perturbations may also degrade the utility of XAI explanations that require further investigation. In this paper, we present an integrated process designed for downstream evaluation tasks, including validating AI model accuracy, evaluating robustness with benchmark perturbations, comparing explanation utility, and assessing overhead. We demonstrate an evaluation scenario involving six computer vision models, which include CNN-based, Transformer-based, and hybrid architectures, three types of perturbations, and five XAI methods, resulting in ninety unique combinations. The process reveals the explanation utility among the XAI methods in terms of the identified key areas responding to the adversarial perturbation. The process produces aggregated results that illustrate multiple attributes of each AI model.

摘要: 随着AI模型的快速发展，它们经常被发布到开放存储库，如HuggingFace。在将这些模型集成到生产开发生命周期之前，对它们执行质量保证验证是至关重要的。除了在准确性和计算成本方面平衡评估效率外，对抗性攻击还对人工智能模型的健壮性和可解释性构成潜在威胁。同时，XAI将近似输入的算法应用于后期输出，以确定贡献特征。对抗性的干扰也可能降低Xai解释的效用，需要进一步的调查。在本文中，我们提出了一个为下游评估任务设计的完整过程，包括验证人工智能模型的准确性，评估基准扰动下的稳健性，比较解释效用，以及评估开销。我们演示了一个包含六个计算机视觉模型的评估场景，其中包括基于CNN的、基于变形金刚的和混合架构、三种类型的扰动和五种XAI方法，产生了90个独特的组合。这个过程揭示了XAI方法之间的解释效用，就识别的关键区域而言，对对抗性扰动的响应。该过程产生的聚合结果说明了每个AI模型的多个属性。



## **16. Secure Aggregation is Not Private Against Membership Inference Attacks**

安全聚合对成员身份推断攻击不是私有的 cs.LG

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.17775v1) [paper-pdf](http://arxiv.org/pdf/2403.17775v1)

**Authors**: Khac-Hoang Ngo, Johan Östman, Giuseppe Durisi, Alexandre Graell i Amat

**Abstract**: Secure aggregation (SecAgg) is a commonly-used privacy-enhancing mechanism in federated learning, affording the server access only to the aggregate of model updates while safeguarding the confidentiality of individual updates. Despite widespread claims regarding SecAgg's privacy-preserving capabilities, a formal analysis of its privacy is lacking, making such presumptions unjustified. In this paper, we delve into the privacy implications of SecAgg by treating it as a local differential privacy (LDP) mechanism for each local update. We design a simple attack wherein an adversarial server seeks to discern which update vector a client submitted, out of two possible ones, in a single training round of federated learning under SecAgg. By conducting privacy auditing, we assess the success probability of this attack and quantify the LDP guarantees provided by SecAgg. Our numerical results unveil that, contrary to prevailing claims, SecAgg offers weak privacy against membership inference attacks even in a single training round. Indeed, it is difficult to hide a local update by adding other independent local updates when the updates are of high dimension. Our findings underscore the imperative for additional privacy-enhancing mechanisms, such as noise injection, in federated learning.

摘要: 安全聚合(SecAgg)是联合学习中常用的隐私增强机制，仅允许服务器访问模型更新的聚合，同时保护单个更新的机密性。尽管人们普遍声称SecAgg具有保护隐私的能力，但缺乏对其隐私的正式分析，这使得这种假设是不合理的。在本文中，我们深入研究了SecAgg的隐私含义，将其视为针对每个本地更新的本地差异隐私(LDP)机制。我们设计了一个简单的攻击，其中敌对服务器试图在SecAgg下的联合学习的单个训练轮中，从两个可能的更新向量中辨别客户端提交的更新向量。通过进行隐私审计，我们评估了该攻击的成功概率，并量化了SecAgg提供的LDP保证。我们的数值结果表明，与流行的说法相反，SecAgg即使在一轮训练中也提供了针对成员推理攻击的弱隐私。事实上，当更新是高维时，很难通过添加其他独立的本地更新来隐藏本地更新。我们的发现强调了联合学习中额外的隐私增强机制的必要性，例如噪音注入。



## **17. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

基于优化的LLM—as—a—Judge快速注入攻击 cs.CR

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.17710v1) [paper-pdf](http://arxiv.org/pdf/2403.17710v1)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge is a novel solution that can assess textual information with large language models (LLMs). Based on existing research studies, LLMs demonstrate remarkable performance in providing a compelling alternative to traditional human assessment. However, the robustness of these systems against prompt injection attacks remains an open question. In this work, we introduce JudgeDeceiver, a novel optimization-based prompt injection attack tailored to LLM-as-a-Judge. Our method formulates a precise optimization objective for attacking the decision-making process of LLM-as-a-Judge and utilizes an optimization algorithm to efficiently automate the generation of adversarial sequences, achieving targeted and effective manipulation of model evaluations. Compared to handcraft prompt injection attacks, our method demonstrates superior efficacy, posing a significant challenge to the current security paradigms of LLM-based judgment systems. Through extensive experiments, we showcase the capability of JudgeDeceiver in altering decision outcomes across various cases, highlighting the vulnerability of LLM-as-a-Judge systems to the optimization-based prompt injection attack.

摘要: LLM-as-a-Court是一种新的解决方案，它可以使用大型语言模型(LLM)来评估文本信息。基于现有的研究，LLMS在提供一种令人信服的替代传统的人类评估方面表现出显著的性能。然而，这些系统对快速注入攻击的健壮性仍然是一个悬而未决的问题。在这项工作中，我们介绍了一种新的基于优化的快速注入攻击，该攻击是针对LLM-as-a-Court定制的。我们的方法为攻击LLM-as-a-Court的决策过程制定了一个精确的优化目标，并利用优化算法高效地自动生成对抗序列，实现了对模型评估的有针对性和有效的操作。与手工即时注入攻击相比，我们的方法表现出更好的有效性，对基于LLM的判断系统的现有安全范例提出了重大挑战。通过大量的实验，我们展示了JudgeDeceiver在改变不同案件的决策结果方面的能力，突出了LLM-as-a-Court系统对基于优化的即时注入攻击的脆弱性。



## **18. Towards more Practical Threat Models in Artificial Intelligence Security**

人工智能安全中更实用的威胁模型 cs.CR

18 pages, 4 figures, 8 tables, accepted to Usenix Security,  incorporated external feedback

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2311.09994v2) [paper-pdf](http://arxiv.org/pdf/2311.09994v2)

**Authors**: Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Alexandre Alahi

**Abstract**: Recent works have identified a gap between research and practice in artificial intelligence security: threats studied in academia do not always reflect the practical use and security risks of AI. For example, while models are often studied in isolation, they form part of larger ML pipelines in practice. Recent works also brought forward that adversarial manipulations introduced by academic attacks are impractical. We take a first step towards describing the full extent of this disparity. To this end, we revisit the threat models of the six most studied attacks in AI security research and match them to AI usage in practice via a survey with 271 industrial practitioners. On the one hand, we find that all existing threat models are indeed applicable. On the other hand, there are significant mismatches: research is often too generous with the attacker, assuming access to information not frequently available in real-world settings. Our paper is thus a call for action to study more practical threat models in artificial intelligence security.

摘要: 最近的研究发现了人工智能安全研究和实践之间的差距：学术界研究的威胁并不总是反映人工智能的实际使用和安全风险。例如，虽然模型通常是孤立研究的，但实际上它们构成了更大的ML管道的一部分。最近的研究也提出，学术攻击引入的对抗性操纵是不切实际的。我们朝着描述这种差距的全面程度迈出了第一步。为此，我们重新审视了人工智能安全研究中研究最多的六种攻击的威胁模型，并通过对271名行业从业者的调查，将它们与人工智能的实际使用相匹配。一方面，我们发现所有现有的威胁模型确实都是适用的。另一方面，存在严重的不匹配：研究往往对攻击者过于慷慨，假设他们可以访问现实世界中不常见的信息。因此，我们的论文呼吁采取行动，研究人工智能安全中更实用的威胁模型。



## **19. FaultGuard: A Generative Approach to Resilient Fault Prediction in Smart Electrical Grids**

FaultGuard：智能电网弹性故障预测的生成方法 cs.CR

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.17494v1) [paper-pdf](http://arxiv.org/pdf/2403.17494v1)

**Authors**: Emad Efatinasab, Francesco Marchiori, Alessandro Brighente, Mirco Rampazzo, Mauro Conti

**Abstract**: Predicting and classifying faults in electricity networks is crucial for uninterrupted provision and keeping maintenance costs at a minimum. Thanks to the advancements in the field provided by the smart grid, several data-driven approaches have been proposed in the literature to tackle fault prediction tasks. Implementing these systems brought several improvements, such as optimal energy consumption and quick restoration. Thus, they have become an essential component of the smart grid. However, the robustness and security of these systems against adversarial attacks have not yet been extensively investigated. These attacks can impair the whole grid and cause additional damage to the infrastructure, deceiving fault detection systems and disrupting restoration. In this paper, we present FaultGuard, the first framework for fault type and zone classification resilient to adversarial attacks. To ensure the security of our system, we employ an Anomaly Detection System (ADS) leveraging a novel Generative Adversarial Network training layer to identify attacks. Furthermore, we propose a low-complexity fault prediction model and an online adversarial training technique to enhance robustness. We comprehensively evaluate the framework's performance against various adversarial attacks using the IEEE13-AdvAttack dataset, which constitutes the state-of-the-art for resilient fault prediction benchmarking. Our model outclasses the state-of-the-art even without considering adversaries, with an accuracy of up to 0.958. Furthermore, our ADS shows attack detection capabilities with an accuracy of up to 1.000. Finally, we demonstrate how our novel training layers drastically increase performances across the whole framework, with a mean increase of 154% in ADS accuracy and 118% in model accuracy.

摘要: 预测和分类电网中的故障对于不间断供电和将维护成本保持在最低水平至关重要。由于智能电网在该领域的进步，文献中已经提出了几种数据驱动的方法来处理故障预测任务。实施这些系统带来了一些改进，例如最佳的能源消耗和快速恢复。因此，它们已成为智能电网的重要组成部分。然而，这些系统对敌意攻击的健壮性和安全性还没有得到广泛的研究。这些攻击可能会损害整个电网，并对基础设施造成额外的破坏，欺骗故障检测系统并中断恢复。在本文中，我们提出了FaultGuard，这是第一个对对手攻击具有弹性的故障类型和区域分类框架。为了确保系统的安全性，我们使用了一个异常检测系统(ADS)，该系统利用一个新的生成性对抗性网络训练层来识别攻击。此外，我们还提出了一种低复杂度的故障预测模型和在线对抗性训练技术来增强鲁棒性。我们使用IEEE13-AdvAttack数据集全面评估了该框架对各种敌意攻击的性能，该数据集构成了弹性故障预测基准测试的最新技术。我们的模型甚至在不考虑对手的情况下也超过了最先进的模型，准确率高达0.958。此外，我们的广告显示了攻击检测能力，准确率高达1.000。最后，我们展示了我们的新型训练层如何显著提高了整个框架的性能，ADS准确率平均提高了154%，模型准确率平均提高了118%。



## **20. Rumor Detection with a novel graph neural network approach**

基于图神经网络的谣言检测方法 cs.AI

10 pages, 5 figures

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.16206v2) [paper-pdf](http://arxiv.org/pdf/2403.16206v2)

**Authors**: Tianrui Liu, Qi Cai, Changxin Xu, Bo Hong, Fanghao Ni, Yuxin Qiao, Tsungwei Yang

**Abstract**: The wide spread of rumors on social media has caused a negative impact on people's daily life, leading to potential panic, fear, and mental health problems for the public. How to debunk rumors as early as possible remains a challenging problem. Existing studies mainly leverage information propagation structure to detect rumors, while very few works focus on correlation among users that they may coordinate to spread rumors in order to gain large popularity. In this paper, we propose a new detection model, that jointly learns both the representations of user correlation and information propagation to detect rumors on social media. Specifically, we leverage graph neural networks to learn the representations of user correlation from a bipartite graph that describes the correlations between users and source tweets, and the representations of information propagation with a tree structure. Then we combine the learned representations from these two modules to classify the rumors. Since malicious users intend to subvert our model after deployment, we further develop a greedy attack scheme to analyze the cost of three adversarial attacks: graph attack, comment attack, and joint attack. Evaluation results on two public datasets illustrate that the proposed MODEL outperforms the state-of-the-art rumor detection models. We also demonstrate our method performs well for early rumor detection. Moreover, the proposed detection method is more robust to adversarial attacks compared to the best existing method. Importantly, we show that it requires a high cost for attackers to subvert user correlation pattern, demonstrating the importance of considering user correlation for rumor detection.

摘要: 谣言在社交媒体上的广泛传播对人们的日常生活造成了负面影响，给公众带来了潜在的恐慌、恐惧和心理健康问题。如何尽早揭穿谣言仍是一个具有挑战性的问题。现有的研究主要是利用信息传播结构来发现谣言，而很少有人关注用户之间的相关性，他们可能会协同传播谣言以获得更大的人气。在本文中，我们提出了一种新的检测模型，该模型同时学习用户相关性和信息传播的表示，以检测社交媒体上的谣言。具体地说，我们利用图神经网络从描述用户和源推文之间的相关性的二部图中学习用户相关性的表示，以及用树结构表示信息传播。然后，我们结合这两个模块的学习表示来对谣言进行分类。由于恶意用户在部署后有意颠覆我们的模型，我们进一步开发了一种贪婪攻击方案，分析了图攻击、评论攻击和联合攻击三种对抗性攻击的代价。在两个公开数据集上的评估结果表明，该模型的性能优于最新的谣言检测模型。我们还证明了我们的方法在早期谣言检测中表现良好。此外，与现有的最佳检测方法相比，本文提出的检测方法对敌意攻击具有更强的鲁棒性。重要的是，我们证明了攻击者要颠覆用户相关性模式需要付出很高的代价，这说明了考虑用户相关性对谣言检测的重要性。



## **21. The Anatomy of Adversarial Attacks: Concept-based XAI Dissection**

对抗性攻击的剖析：基于概念的XAI解剖 cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16782v1) [paper-pdf](http://arxiv.org/pdf/2403.16782v1)

**Authors**: Georgii Mikriukov, Gesina Schwalbe, Franz Motzkus, Korinna Bade

**Abstract**: Adversarial attacks (AAs) pose a significant threat to the reliability and robustness of deep neural networks. While the impact of these attacks on model predictions has been extensively studied, their effect on the learned representations and concepts within these models remains largely unexplored. In this work, we perform an in-depth analysis of the influence of AAs on the concepts learned by convolutional neural networks (CNNs) using eXplainable artificial intelligence (XAI) techniques. Through an extensive set of experiments across various network architectures and targeted AA techniques, we unveil several key findings. First, AAs induce substantial alterations in the concept composition within the feature space, introducing new concepts or modifying existing ones. Second, the adversarial perturbation itself can be linearly decomposed into a set of latent vector components, with a subset of these being responsible for the attack's success. Notably, we discover that these components are target-specific, i.e., are similar for a given target class throughout different AA techniques and starting classes. Our findings provide valuable insights into the nature of AAs and their impact on learned representations, paving the way for the development of more robust and interpretable deep learning models, as well as effective defenses against adversarial threats.

摘要: 对抗性攻击(AAs)对深度神经网络的可靠性和健壮性构成了严重威胁。虽然这些攻击对模型预测的影响已经被广泛研究，但它们对这些模型中的学习表示和概念的影响在很大程度上仍未被探索。在这项工作中，我们使用可解释人工智能(XAI)技术深入分析了人工智能对卷积神经网络(CNN)学习的概念的影响。通过跨各种网络架构和有针对性的AA技术进行的一组广泛的实验，我们揭示了几个关键发现。首先，人工智能在特征空间内引起概念构成的实质性变化，引入新的概念或修改现有的概念。其次，敌方扰动本身可以线性分解为一组潜在向量分量，这些分量的子集是攻击成功的原因。值得注意的是，我们发现这些组件是特定于目标的，即在不同的AA技术和起始类中，对于给定的目标类是相似的。我们的发现对人工智能的性质及其对学习陈述的影响提供了有价值的见解，为开发更健壮和可解释的深度学习模型以及有效防御对手威胁铺平了道路。



## **22. DeepKnowledge: Generalisation-Driven Deep Learning Testing**

DeepKnowledge：泛化驱动的深度学习测试 cs.LG

10 pages

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16768v1) [paper-pdf](http://arxiv.org/pdf/2403.16768v1)

**Authors**: Sondess Missaoui, Simos Gerasimou, Nikolaos Matragkas

**Abstract**: Despite their unprecedented success, DNNs are notoriously fragile to small shifts in data distribution, demanding effective testing techniques that can assess their dependability. Despite recent advances in DNN testing, there is a lack of systematic testing approaches that assess the DNN's capability to generalise and operate comparably beyond data in their training distribution. We address this gap with DeepKnowledge, a systematic testing methodology for DNN-based systems founded on the theory of knowledge generalisation, which aims to enhance DNN robustness and reduce the residual risk of 'black box' models. Conforming to this theory, DeepKnowledge posits that core computational DNN units, termed Transfer Knowledge neurons, can generalise under domain shift. DeepKnowledge provides an objective confidence measurement on testing activities of DNN given data distribution shifts and uses this information to instrument a generalisation-informed test adequacy criterion to check the transfer knowledge capacity of a test set. Our empirical evaluation of several DNNs, across multiple datasets and state-of-the-art adversarial generation techniques demonstrates the usefulness and effectiveness of DeepKnowledge and its ability to support the engineering of more dependable DNNs. We report improvements of up to 10 percentage points over state-of-the-art coverage criteria for detecting adversarial attacks on several benchmarks, including MNIST, SVHN, and CIFAR.

摘要: 尽管DNN取得了前所未有的成功，但众所周知，它们对数据分布的微小变化很脆弱，需要有效的测试技术来评估它们的可靠性。尽管最近在DNN测试方面取得了进展，但缺乏系统的测试方法来评估DNN在其训练分布中的数据之外的泛化和相对操作的能力。我们用DeepKnowledge解决了这一差距，DeepKnowledge是一种基于知识泛化理论的DNN系统测试方法，旨在增强DNN的健壮性并降低“黑盒”模型的残余风险。与这一理论相一致的是，DeepKnowledge提出，核心计算DNN单元，称为传递知识神经元，可以在域转移下泛化。DeepKnowledge在给定数据分布漂移的情况下对DNN的测试活动提供了客观的置信度度量，并使用该信息来测试泛化信息的测试充分性标准，以检查测试集的传递知识能力。我们对几个DNN的经验评估，跨越多个数据集和最先进的对手生成技术，证明了DeepKnowledge的有用性和有效性，以及它支持设计更可靠的DNN的能力。我们报告了在包括MNIST、SVHN和CIFAR在内的几个基准上检测对抗性攻击的最新覆盖标准的改进，提高了高达10个百分点。



## **23. Boosting Adversarial Transferability by Block Shuffle and Rotation**

利用块洗牌和旋转增强对抗性传递 cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2308.10299v3) [paper-pdf](http://arxiv.org/pdf/2308.10299v3)

**Authors**: Kunyu Wang, Xuanran He, Wenxuan Wang, Xiaosen Wang

**Abstract**: Adversarial examples mislead deep neural networks with imperceptible perturbations and have brought significant threats to deep learning. An important aspect is their transferability, which refers to their ability to deceive other models, thus enabling attacks in the black-box setting. Though various methods have been proposed to boost transferability, the performance still falls short compared with white-box attacks. In this work, we observe that existing input transformation based attacks, one of the mainstream transfer-based attacks, result in different attention heatmaps on various models, which might limit the transferability. We also find that breaking the intrinsic relation of the image can disrupt the attention heatmap of the original image. Based on this finding, we propose a novel input transformation based attack called block shuffle and rotation (BSR). Specifically, BSR splits the input image into several blocks, then randomly shuffles and rotates these blocks to construct a set of new images for gradient calculation. Empirical evaluations on the ImageNet dataset demonstrate that BSR could achieve significantly better transferability than the existing input transformation based methods under single-model and ensemble-model settings. Combining BSR with the current input transformation method can further improve the transferability, which significantly outperforms the state-of-the-art methods. Code is available at https://github.com/Trustworthy-AI-Group/BSR

摘要: 对抗性例子用潜移默化的扰动误导了深度神经网络，给深度学习带来了重大威胁。一个重要的方面是它们的可转移性，这指的是它们欺骗其他模型的能力，从而使攻击能够在黑盒环境中进行。虽然已经提出了各种方法来提高可转移性，但与白盒攻击相比，性能仍然不足。在这项工作中，我们观察到现有的基于输入变换的攻击是基于转移的主流攻击之一，在不同的模型上会导致不同的注意力热图，这可能会限制可转移性。我们还发现，打破图像的内在联系会扰乱原始图像的注意热图。基于这一发现，我们提出了一种新的基于输入变换的攻击方法，称为块置乱和旋转攻击(BSR)。具体地说，BSR将输入图像分成几个块，然后随机地对这些块进行洗牌和旋转，以构建一组新的图像用于梯度计算。在ImageNet数据集上的实证评估表明，在单模型和集成模型的设置下，BSR可以获得比现有的基于输入变换的方法更好的可转移性。将BSR与当前的输入变换方法相结合，可以进一步提高可转移性，显著优于最先进的方法。代码可在https://github.com/Trustworthy-AI-Group/BSR上找到



## **24. A Huber Loss Minimization Approach to Byzantine Robust Federated Learning**

拜占庭鲁棒联邦学习的Huber损失最小化方法 cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2308.12581v2) [paper-pdf](http://arxiv.org/pdf/2308.12581v2)

**Authors**: Puning Zhao, Fei Yu, Zhiguo Wan

**Abstract**: Federated learning systems are susceptible to adversarial attacks. To combat this, we introduce a novel aggregator based on Huber loss minimization, and provide a comprehensive theoretical analysis. Under independent and identically distributed (i.i.d) assumption, our approach has several advantages compared to existing methods. Firstly, it has optimal dependence on $\epsilon$, which stands for the ratio of attacked clients. Secondly, our approach does not need precise knowledge of $\epsilon$. Thirdly, it allows different clients to have unequal data sizes. We then broaden our analysis to include non-i.i.d data, such that clients have slightly different distributions.

摘要: 联邦学习系统容易受到对抗性攻击。为了解决这一问题，我们提出了一种新的基于Huber损失最小化的聚合器，并提供了全面的理论分析。在独立同分布（i.i. d）假设下，我们的方法与现有方法相比具有几个优点。首先，它对$\rn $具有最优依赖性，它代表受攻击客户端的比率。第二，我们的方法不需要精确的$\rn $知识。第三，它允许不同的客户端具有不相等的数据大小。然后，我们扩大我们的分析范围，包括非i.i. d数据，这样客户的分布略有不同。



## **25. Deciphering the Interplay between Local Differential Privacy, Average Bayesian Privacy, and Maximum Bayesian Privacy**

解密局部差分隐私、平均贝叶斯隐私和最大贝叶斯隐私之间的相互作用 cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16591v1) [paper-pdf](http://arxiv.org/pdf/2403.16591v1)

**Authors**: Xiaojin Zhang, Yulin Fei, Wei Chen, Hai Jin

**Abstract**: The swift evolution of machine learning has led to emergence of various definitions of privacy due to the threats it poses to privacy, including the concept of local differential privacy (LDP). Although widely embraced and utilized across numerous domains, this conventional approach to measure privacy still exhibits certain limitations, spanning from failure to prevent inferential disclosure to lack of consideration for the adversary's background knowledge. In this comprehensive study, we introduce Bayesian privacy and delve into the intricate relationship between local differential privacy and its Bayesian counterparts, unveiling novel insights into utility-privacy trade-offs. We introduce a framework that encapsulates both attack and defense strategies, highlighting their interplay and effectiveness. Our theoretical contributions are anchored in the rigorous definitions and relationships between Average Bayesian Privacy (ABP) and Maximum Bayesian Privacy (MBP), encapsulated by equations $\epsilon_{p,a} \leq \frac{1}{\sqrt{2}}\sqrt{(\epsilon_{p,m} + \epsilon)\cdot(e^{\epsilon_{p,m} + \epsilon} - 1)}$ and the equivalence between $\xi$-MBP and $2\xi$-LDP established under uniform prior distribution. These relationships fortify our understanding of the privacy guarantees provided by various mechanisms, leading to the realization that a mechanism satisfying $\xi$-LDP also confers $\xi$-MBP, and vice versa. Our work not only lays the groundwork for future empirical exploration but also promises to enhance the design of privacy-preserving algorithms that do not compromise on utility, thereby fostering the development of trustworthy machine learning solutions.

摘要: 机器学习的快速发展导致了各种隐私定义的出现，因为它对隐私构成了威胁，包括局部差异隐私(LDP)的概念。尽管这种衡量隐私的传统方法在许多领域得到了广泛的接受和应用，但它仍然显示出一定的局限性，从未能阻止推论披露到缺乏对对手背景知识的考虑。在这项全面的研究中，我们引入了贝叶斯隐私，并深入研究了局部差异隐私与其对应的贝叶斯隐私之间的复杂关系，揭示了对效用-隐私权衡的新见解。我们引入了一个框架，该框架封装了攻击和防御战略，突出了它们的相互作用和有效性。我们的理论贡献植根于平均贝叶斯隐私度和最大贝叶斯隐私度之间的严格定义和关系，以及在均匀先验分布下建立的方程$\epsilon_{p，a}\leq\frac{1}{\sqrt{2}}\Sqrt{(\epsilon_{p，m}+\epsilon)\cdot(e^{\epsilon_{p，m}+\epsilon}-1)}}以及$\xi$-mBP和$2\xi$-ldp之间的等价性。这些关系加强了我们对各种机制提供的隐私保障的理解，导致我们认识到满足$\xi$-LDP的机制也授予$\xi$-MBP，反之亦然。我们的工作不仅为未来的经验探索奠定了基础，而且承诺加强隐私保护算法的设计，从而促进可信机器学习解决方案的发展。



## **26. Revealing Vulnerabilities of Neural Networks in Parameter Learning and Defense Against Explanation-Aware Backdoors**

揭示神经网络参数学习中的漏洞及防范知识后门 cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16569v1) [paper-pdf](http://arxiv.org/pdf/2403.16569v1)

**Authors**: Md Abdul Kadir, GowthamKrishna Addluri, Daniel Sonntag

**Abstract**: Explainable Artificial Intelligence (XAI) strategies play a crucial part in increasing the understanding and trustworthiness of neural networks. Nonetheless, these techniques could potentially generate misleading explanations. Blinding attacks can drastically alter a machine learning algorithm's prediction and explanation, providing misleading information by adding visually unnoticeable artifacts into the input, while maintaining the model's accuracy. It poses a serious challenge in ensuring the reliability of XAI methods. To ensure the reliability of XAI methods poses a real challenge, we leverage statistical analysis to highlight the changes in CNN weights within a CNN following blinding attacks. We introduce a method specifically designed to limit the effectiveness of such attacks during the evaluation phase, avoiding the need for extra training. The method we suggest defences against most modern explanation-aware adversarial attacks, achieving an approximate decrease of ~99\% in the Attack Success Rate (ASR) and a ~91\% reduction in the Mean Square Error (MSE) between the original explanation and the defended (post-attack) explanation across three unique types of attacks.

摘要: 可解释人工智能(XAI)策略在增加神经网络的理解和可信度方面发挥着至关重要的作用。然而，这些技术可能会产生误导性的解释。盲目攻击可以极大地改变机器学习算法的预测和解释，通过在输入中添加视觉上不可察觉的伪像来提供误导性信息，同时保持模型的准确性。这对确保XAI方法的可靠性提出了严峻的挑战。为了确保XAI方法的可靠性构成了一个真正的挑战，我们利用统计分析来突出CNN在盲人攻击后CNN权重的变化。我们引入了一种专门设计的方法，在评估阶段限制此类攻击的有效性，避免了额外培训的需要。我们提出的方法可以防御大多数现代解释感知的对手攻击，在三种独特的攻击类型中，攻击成功率(ASR)大约降低了99%，原始解释和防御(攻击后)解释之间的均方误差(MSE)降低了约91%。



## **27. Exploring the Adversarial Capabilities of Large Language Models**

探索大型语言模型的对抗能力 cs.AI

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2402.09132v3) [paper-pdf](http://arxiv.org/pdf/2402.09132v3)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.

摘要: 大型语言模型因其强大的语言生成能力而引起了广泛的关注，为工业和研究提供了巨大的潜力。虽然之前的研究已经深入研究了LLMS的安全和隐私问题，但这些模型在多大程度上可以表现出敌对行为，仍然很大程度上还没有被探索。针对这一差距，我们调查了常见的公开可用的LLM是否具有固有的能力来扰乱文本样本以愚弄安全措施，即所谓的对抗性示例攻击。更具体地说，我们调查LLM是否天生就能够从良性样本中制作敌意示例，以愚弄现有的安全Rail。我们的实验集中在仇恨语音检测上，实验表明，LLMS成功地发现了敌意扰动，有效地破坏了仇恨语音检测系统。我们的发现对依赖LLMS的(半)自治系统具有重大影响，突显了它们与现有系统和安全措施相互作用的潜在挑战。



## **28. On the resilience of Collaborative Learning-based Recommender Systems Against Community Detection Attack**

基于协作学习的推荐系统对社区检测攻击的抵抗能力研究 cs.IR

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2306.08929v2) [paper-pdf](http://arxiv.org/pdf/2306.08929v2)

**Authors**: Yacine Belal, Sonia Ben Mokhtar, Mohamed Maouche, Anthony Simonet-Boulogne

**Abstract**: Collaborative-learning-based recommender systems emerged following the success of collaborative learning techniques such as Federated Learning (FL) and Gossip Learning (GL). In these systems, users participate in the training of a recommender system while maintaining their history of consumed items on their devices. While these solutions seemed appealing for preserving the privacy of the participants at first glance, recent studies have revealed that collaborative learning can be vulnerable to various privacy attacks. In this paper, we study the resilience of collaborative learning-based recommender systems against a novel privacy attack called Community Detection Attack (CDA). This attack enables an adversary to identify community members based on a chosen set of items (eg., identifying users interested in specific points-of-interest). Through experiments on three real recommendation datasets using two state-of-the-art recommendation models, we evaluate the sensitivity of an FL-based recommender system as well as two flavors of Gossip Learning-based recommender systems to CDA. The results show that across all models and datasets, the FL setting is more vulnerable to CDA compared to Gossip settings. Furthermore, we assess two off-the-shelf mitigation strategies, namely differential privacy (DP) and a \emph{Share less} policy, which consists of sharing a subset of less sensitive model parameters. The findings indicate a more favorable privacy-utility trade-off for the \emph{Share less} strategy, particularly in FedRecs.

摘要: 基于协作学习的推荐系统是在联邦学习(FL)和八卦学习(GL)等协作学习技术成功之后应运而生的。在这些系统中，用户参与推荐系统的培训，同时在他们的设备上维护他们的消费项目的历史。乍一看，这些解决方案在保护参与者隐私方面似乎很有吸引力，但最近的研究表明，协作学习可能容易受到各种隐私攻击。本文研究了基于协作学习的推荐系统对一种新的隐私攻击--社区检测攻击(CDA)的恢复能力。这种攻击使对手能够根据选定的一组项目识别社区成员(例如，识别对特定兴趣点感兴趣的用户)。通过使用两种最新推荐模型在三个真实推荐数据集上的实验，我们评估了一个基于FL的推荐系统以及两种基于八卦学习的推荐系统对CDA的敏感度。结果表明，在所有模型和数据集中，与八卦设置相比，FL设置更容易受到CDA的影响。此外，我们评估了两种现成的缓解策略，即差异隐私(DP)和共享较少的策略，该策略包括共享不太敏感的模型参数的子集。研究结果表明，EMPH{Share Less}策略的隐私效用权衡更有利，尤其是在FedRecs中。



## **29. Model-less Is the Best Model: Generating Pure Code Implementations to Replace On-Device DL Models**

无模型是最好的模型：生成纯代码实现来替换设备上的DL模型 cs.SE

Accepted by the ACM SIGSOFT International Symposium on Software  Testing and Analysis (ISSTA2024)

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16479v1) [paper-pdf](http://arxiv.org/pdf/2403.16479v1)

**Authors**: Mingyi Zhou, Xiang Gao, Pei Liu, John Grundy, Chunyang Chen, Xiao Chen, Li Li

**Abstract**: Recent studies show that deployed deep learning (DL) models such as those of Tensor Flow Lite (TFLite) can be easily extracted from real-world applications and devices by attackers to generate many kinds of attacks like adversarial attacks. Although securing deployed on-device DL models has gained increasing attention, no existing methods can fully prevent the aforementioned threats. Traditional software protection techniques have been widely explored, if on-device models can be implemented using pure code, such as C++, it will open the possibility of reusing existing software protection techniques. However, due to the complexity of DL models, there is no automatic method that can translate the DL models to pure code. To fill this gap, we propose a novel method, CustomDLCoder, to automatically extract the on-device model information and synthesize a customized executable program for a wide range of DL models. CustomDLCoder first parses the DL model, extracts its backend computing units, configures the computing units to a graph, and then generates customized code to implement and deploy the ML solution without explicit model representation. The synthesized program hides model information for DL deployment environments since it does not need to retain explicit model representation, preventing many attacks on the DL model. In addition, it improves ML performance because the customized code removes model parsing and preprocessing steps and only retains the data computing process. Our experimental results show that CustomDLCoder improves model security by disabling on-device model sniffing. Compared with the original on-device platform (i.e., TFLite), our method can accelerate model inference by 21.0% and 24.3% on x86-64 and ARM64 platforms, respectively. Most importantly, it can significantly reduce memory consumption by 68.8% and 36.0% on x86-64 and ARM64 platforms, respectively.

摘要: 最近的研究表明，部署的深度学习(DL)模型，如张量流精简(TFLite)模型，可以很容易地被攻击者从现实世界的应用和设备中提取出来，从而产生多种攻击，如对抗性攻击。尽管保护部署在设备上的DL模型越来越受到关注，但没有一种现有方法可以完全防止上述威胁。传统的软件保护技术已经得到了广泛的探索，如果设备上的模型可以用纯代码实现，如C++，这将打开重用现有软件保护技术的可能性。然而，由于DL模型的复杂性，目前还没有一种自动的方法可以将DL模型转换为纯代码。为了填补这一空白，我们提出了一种新的方法CustomDLCoder，它可以自动提取设备上的模型信息，并为广泛的DL模型合成定制的可执行程序。CustomDLCoder首先解析DL模型，提取其后端计算单元，将计算单元配置为图形，然后生成定制代码来实现和部署ML解决方案，而不需要显式的模型表示。合成的程序隐藏了DL部署环境的模型信息，因为它不需要保留显式的模型表示，从而防止了对DL模型的许多攻击。此外，它还提高了ML的性能，因为定制的代码删除了模型解析和预处理步骤，只保留了数据计算过程。我们的实验结果表明，CustomDLCoder通过禁止设备上的模型嗅探提高了模型的安全性。在x86-64和ARM64平台上，与原有的设备上平台(即TFLite)相比，该方法的模型推理速度分别提高了21.0%和24.3%。最重要的是，它可以在x86-64和ARM64平台上分别显著降低68.8%和36.0%的内存消耗。



## **30. Secure Control of Connected and Automated Vehicles Using Trust-Aware Robust Event-Triggered Control Barrier Functions**

使用信任感知鲁棒事件触发控制屏障函数的联网和自动车辆安全控制 eess.SY

arXiv admin note: substantial text overlap with arXiv:2305.16818

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2401.02306v3) [paper-pdf](http://arxiv.org/pdf/2401.02306v3)

**Authors**: H M Sabbir Ahmad, Ehsan Sabouni, Akua Dickson, Wei Xiao, Christos G. Cassandras, Wenchao Li

**Abstract**: We address the security of a network of Connected and Automated Vehicles (CAVs) cooperating to safely navigate through a conflict area (e.g., traffic intersections, merging roadways, roundabouts). Previous studies have shown that such a network can be targeted by adversarial attacks causing traffic jams or safety violations ending in collisions. We focus on attacks targeting the V2X communication network used to share vehicle data and consider as well uncertainties due to noise in sensor measurements and communication channels. To combat these, motivated by recent work on the safe control of CAVs, we propose a trust-aware robust event-triggered decentralized control and coordination framework that can provably guarantee safety. We maintain a trust metric for each vehicle in the network computed based on their behavior and used to balance the tradeoff between conservativeness (when deeming every vehicle as untrustworthy) and guaranteed safety and security. It is important to highlight that our framework is invariant to the specific choice of the trust framework. Based on this framework, we propose an attack detection and mitigation scheme which has twofold benefits: (i) the trust framework is immune to false positives, and (ii) it provably guarantees safety against false positive cases. We use extensive simulations (in SUMO and CARLA) to validate the theoretical guarantees and demonstrate the efficacy of our proposed scheme to detect and mitigate adversarial attacks.

摘要: 我们致力于解决互联和自动化车辆(CAV)网络的安全问题，这些车辆通过协作安全地通过冲突区域(例如，交通路口、合并道路、环形交叉路口)。以前的研究表明，这样的网络可以成为导致交通拥堵或以碰撞结束的安全违规行为的对抗性攻击的目标。我们专注于针对用于共享车辆数据的V2X通信网络的攻击，并考虑由于传感器测量和通信通道中的噪声而产生的不确定性。为了应对这些问题，基于最近在CAV安全控制方面的工作，我们提出了一个信任感知的、健壮的、事件触发的分布式控制和协调框架，该框架能够有效地保证安全。我们为网络中的每辆车维护一个基于其行为计算的信任度量，用于平衡保守性(当认为每辆车不值得信任时)与保证的安全和保障之间的权衡。必须强调的是，我们的框架与信任框架的具体选择是不变的。基于该框架，我们提出了一种攻击检测和缓解方案，该方案具有两个优点：(I)信任框架不受误报的影响；(Ii)它可证明地保证了对误报情况的安全性。我们使用大量的仿真(在相扑和CALA中)来验证理论上的保证，并展示了我们所提出的方案在检测和缓解敌意攻击方面的有效性。



## **31. Ensemble Adversarial Defense via Integration of Multiple Dispersed Low Curvature Models**

多离散低曲率模型集成对抗防御 cs.LG

Accepted to The 2024 International Joint Conference on Neural  Networks (IJCNN)

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16405v1) [paper-pdf](http://arxiv.org/pdf/2403.16405v1)

**Authors**: Kaikang Zhao, Xi Chen, Wei Huang, Liuxin Ding, Xianglong Kong, Fan Zhang

**Abstract**: The integration of an ensemble of deep learning models has been extensively explored to enhance defense against adversarial attacks. The diversity among sub-models increases the attack cost required to deceive the majority of the ensemble, thereby improving the adversarial robustness. While existing approaches mainly center on increasing diversity in feature representations or dispersion of first-order gradients with respect to input, the limited correlation between these diversity metrics and adversarial robustness constrains the performance of ensemble adversarial defense. In this work, we aim to enhance ensemble diversity by reducing attack transferability. We identify second-order gradients, which depict the loss curvature, as a key factor in adversarial robustness. Computing the Hessian matrix involved in second-order gradients is computationally expensive. To address this, we approximate the Hessian-vector product using differential approximation. Given that low curvature provides better robustness, our ensemble model was designed to consider the influence of curvature among different sub-models. We introduce a novel regularizer to train multiple more-diverse low-curvature network models. Extensive experiments across various datasets demonstrate that our ensemble model exhibits superior robustness against a range of attacks, underscoring the effectiveness of our approach.

摘要: 集成深度学习模型已被广泛探索，以增强对对手攻击的防御。子模型之间的多样性增加了欺骗大部分集成所需的攻击成本，从而提高了对手的稳健性。虽然现有的方法主要集中在增加特征表示的多样性或关于输入的一阶梯度的离散，但这些多样性度量与对抗稳健性之间的有限相关性限制了集成对抗防御的性能。在这项工作中，我们的目标是通过降低攻击的可转移性来提高集合的多样性。我们认为描述损失曲率的二阶梯度是对抗健壮性的一个关键因素。计算二阶梯度所涉及的海森矩阵在计算上是昂贵的。为了解决这个问题，我们使用微分近似来近似黑森向量积。考虑到低曲率提供了更好的稳健性，我们的集成模型被设计为考虑曲率在不同子模型之间的影响。我们引入了一种新的正则化方法来训练多个更多样化的低曲率网络模型。在不同数据集上的广泛实验表明，我们的集成模型对一系列攻击表现出了卓越的鲁棒性，强调了我们方法的有效性。



## **32. Generating Potent Poisons and Backdoors from Scratch with Guided Diffusion**

利用引导扩散从划痕中产生潜在毒药和后门 cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16365v1) [paper-pdf](http://arxiv.org/pdf/2403.16365v1)

**Authors**: Hossein Souri, Arpit Bansal, Hamid Kazemi, Liam Fowl, Aniruddha Saha, Jonas Geiping, Andrew Gordon Wilson, Rama Chellappa, Tom Goldstein, Micah Goldblum

**Abstract**: Modern neural networks are often trained on massive datasets that are web scraped with minimal human inspection. As a result of this insecure curation pipeline, an adversary can poison or backdoor the resulting model by uploading malicious data to the internet and waiting for a victim to scrape and train on it. Existing approaches for creating poisons and backdoors start with randomly sampled clean data, called base samples, and then modify those samples to craft poisons. However, some base samples may be significantly more amenable to poisoning than others. As a result, we may be able to craft more potent poisons by carefully choosing the base samples. In this work, we use guided diffusion to synthesize base samples from scratch that lead to significantly more potent poisons and backdoors than previous state-of-the-art attacks. Our Guided Diffusion Poisoning (GDP) base samples can be combined with any downstream poisoning or backdoor attack to boost its effectiveness. Our implementation code is publicly available at: https://github.com/hsouri/GDP .

摘要: 现代神经网络通常是在海量数据集上进行训练的，这些数据集是在最少的人类检查的情况下从网络上刮下来的。由于这种不安全的管理管道，对手可以通过将恶意数据上传到互联网并等待受害者对其进行擦除和训练来毒害或后门生成的模型。现有的制造毒药和后门的方法从随机抽样的清洁数据开始，称为基础样本，然后修改这些样本以制造毒药。然而，一些碱基样品可能比其他样品更容易中毒。因此，通过仔细选择基础样品，我们可能能够制造出更强的毒药。在这项工作中，我们使用引导扩散来从头开始合成基本样本，这些样本导致的毒药和后门比以前最先进的攻击要强得多。我们的引导式扩散中毒(GDP)基础样本可以与任何下游中毒或后门攻击相结合，以提高其有效性。我们的实现代码可在https://github.com/hsouri/GDP上公开获得。



## **33. Subspace Defense: Discarding Adversarial Perturbations by Learning a Subspace for Clean Signals**

子空间防御：通过学习干净信号的子空间来丢弃对抗扰动 cs.LG

Accepted by COLING 2024

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2403.16176v1) [paper-pdf](http://arxiv.org/pdf/2403.16176v1)

**Authors**: Rui Zheng, Yuhao Zhou, Zhiheng Xi, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: Deep neural networks (DNNs) are notoriously vulnerable to adversarial attacks that place carefully crafted perturbations on normal examples to fool DNNs. To better understand such attacks, a characterization of the features carried by adversarial examples is needed. In this paper, we tackle this challenge by inspecting the subspaces of sample features through spectral analysis. We first empirically show that the features of either clean signals or adversarial perturbations are redundant and span in low-dimensional linear subspaces respectively with minimal overlap, and the classical low-dimensional subspace projection can suppress perturbation features out of the subspace of clean signals. This makes it possible for DNNs to learn a subspace where only features of clean signals exist while those of perturbations are discarded, which can facilitate the distinction of adversarial examples. To prevent the residual perturbations that is inevitable in subspace learning, we propose an independence criterion to disentangle clean signals from perturbations. Experimental results show that the proposed strategy enables the model to inherently suppress adversaries, which not only boosts model robustness but also motivates new directions of effective adversarial defense.

摘要: 众所周知，深度神经网络(DNN)容易受到敌意攻击，这些攻击会对正常示例进行精心设计的扰动，以愚弄DNN。为了更好地理解这类攻击，需要对对抗性例子所具有的特征进行描述。在本文中，我们通过谱分析检查样本特征的子空间来应对这一挑战。我们首先从经验上证明了CLEAN信号和对抗性扰动的特征在低维线性子空间中是冗余的且重叠最小，经典的低维子空间投影可以抑制CLEAN信号子空间之外的扰动特征。这使得DNN能够学习一个子空间，其中只有干净信号的特征存在，而扰动的特征被丢弃，这有助于区分对抗性例子。为了防止子空间学习中不可避免的残留扰动，我们提出了一个独立准则来分离干净的信号和扰动。实验结果表明，该策略使模型具有内在的抑制能力，不仅增强了模型的稳健性，而且为有效的对抗防御提供了新的方向。



## **34. Large Language Models for Blockchain Security: A Systematic Literature Review**

区块链安全的大型语言模型：系统文献综述 cs.CR

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2403.14280v2) [paper-pdf](http://arxiv.org/pdf/2403.14280v2)

**Authors**: Zheyuan He, Zihao Li, Sen Yang

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools in various domains involving blockchain security (BS). Several recent studies are exploring LLMs applied to BS. However, there remains a gap in our understanding regarding the full scope of applications, impacts, and potential constraints of LLMs on blockchain security. To fill this gap, we conduct a literature review on LLM4BS.   As the first review of LLM's application on blockchain security, our study aims to comprehensively analyze existing research and elucidate how LLMs contribute to enhancing the security of blockchain systems. Through a thorough examination of scholarly works, we delve into the integration of LLMs into various aspects of blockchain security. We explore the mechanisms through which LLMs can bolster blockchain security, including their applications in smart contract auditing, identity verification, anomaly detection, vulnerable repair, and so on. Furthermore, we critically assess the challenges and limitations associated with leveraging LLMs for blockchain security, considering factors such as scalability, privacy concerns, and adversarial attacks. Our review sheds light on the opportunities and potential risks inherent in this convergence, providing valuable insights for researchers, practitioners, and policymakers alike.

摘要: 大型语言模型(LLM)在涉及区块链安全(BS)的各个领域中已成为强大的工具。最近的几项研究正在探索将LLMS应用于BS。然而，对于低成本管理的全部应用范围、影响以及对区块链安全的潜在限制，我们的理解仍然存在差距。为了填补这一空白，我们对LLM4BS进行了文献综述。作为LLM在区块链安全方面应用的首次综述，本研究旨在全面分析现有研究，阐明LLM如何为增强区块链系统的安全性做出贡献。通过对学术著作的深入研究，我们深入研究了LLMS在区块链安全的各个方面的整合。我们探讨了LLMS增强区块链安全的机制，包括它们在智能合同审计、身份验证、异常检测、漏洞修复等方面的应用。此外，考虑到可扩展性、隐私问题和敌意攻击等因素，我们严格评估了利用LLM实现区块链安全所面临的挑战和限制。我们的审查揭示了这种融合所固有的机遇和潜在风险，为研究人员、从业者和政策制定者提供了有价值的见解。



## **35. ALI-DPFL: Differentially Private Federated Learning with Adaptive Local Iterations**

ALI—DPFL：具有自适应局部迭代的差分私有联邦学习 cs.LG

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2308.10457v5) [paper-pdf](http://arxiv.org/pdf/2308.10457v5)

**Authors**: Xinpeng Ling, Jie Fu, Kuncan Wang, Haitao Liu, Zhili Chen

**Abstract**: Federated Learning (FL) is a distributed machine learning technique that allows model training among multiple devices or organizations by sharing training parameters instead of raw data. However, adversaries can still infer individual information through inference attacks (e.g. differential attacks) on these training parameters. As a result, Differential Privacy (DP) has been widely used in FL to prevent such attacks.   We consider differentially private federated learning in a resource-constrained scenario, where both privacy budget and communication rounds are constrained. By theoretically analyzing the convergence, we can find the optimal number of local DPSGD iterations for clients between any two sequential global updates. Based on this, we design an algorithm of Differentially Private Federated Learning with Adaptive Local Iterations (ALI-DPFL). We experiment our algorithm on the MNIST, FashionMNIST and Cifar10 datasets, and demonstrate significantly better performances than previous work in the resource-constraint scenario. Code is available at https://github.com/KnightWan/ALI-DPFL.

摘要: 联合学习(FL)是一种分布式机器学习技术，通过共享训练参数而不是原始数据，允许在多个设备或组织之间进行模型训练。然而，攻击者仍然可以通过对这些训练参数的推理攻击(例如差异攻击)来推断个人信息。因此，差分隐私(DP)被广泛应用于FL中以防止此类攻击。我们考虑在资源受限的情况下进行不同的私有联合学习，其中隐私预算和通信回合都受到限制。通过对收敛的理论分析，我们可以找到任意两个连续全局更新之间客户端的最优局部DPSGD迭代次数。在此基础上，设计了一种基于自适应局部迭代的差分私有联邦学习算法(ALI-DPFL)。我们在MNIST、FashionMNIST和Cifar10数据集上测试了我们的算法，并在资源受限的情况下展示了比以前的工作更好的性能。代码可在https://github.com/KnightWan/ALI-DPFL.上找到



## **36. Robust Diffusion Models for Adversarial Purification**

对抗净化的鲁棒扩散模型 cs.CV

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2403.16067v1) [paper-pdf](http://arxiv.org/pdf/2403.16067v1)

**Authors**: Guang Lin, Zerui Tao, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Diffusion models (DMs) based adversarial purification (AP) has shown to be the most powerful alternative to adversarial training (AT). However, these methods neglect the fact that pre-trained diffusion models themselves are not robust to adversarial attacks as well. Additionally, the diffusion process can easily destroy semantic information and generate a high quality image but totally different from the original input image after the reverse process, leading to degraded standard accuracy. To overcome these issues, a natural idea is to harness adversarial training strategy to retrain or fine-tune the pre-trained diffusion model, which is computationally prohibitive. We propose a novel robust reverse process with adversarial guidance, which is independent of given pre-trained DMs and avoids retraining or fine-tuning the DMs. This robust guidance can not only ensure to generate purified examples retaining more semantic content but also mitigate the accuracy-robustness trade-off of DMs for the first time, which also provides DM-based AP an efficient adaptive ability to new attacks. Extensive experiments are conducted to demonstrate that our method achieves the state-of-the-art results and exhibits generalization against different attacks.

摘要: 基于扩散模型(DM)的对抗净化(AP)已被证明是对抗训练(AT)最有效的替代方法。然而，这些方法忽略了这样一个事实，即预先训练的扩散模型本身对对手攻击也不是很健壮。此外，扩散过程容易破坏语义信息，生成高质量的图像，但反向处理后的图像与原始输入图像完全不同，导致标准精度下降。为了克服这些问题，一个自然的想法是利用对抗性训练策略来重新训练或微调预先训练的扩散模型，这在计算上是令人望而却步的。我们提出了一种新的具有对抗性指导的稳健逆向过程，它独立于给定的预先训练的DM，并且避免了对DM的重新训练或微调。这种健壮的指导不仅可以确保生成保持更多语义内容的纯化实例，还可以第一次缓解DM的准确性和健壮性之间的权衡，这也为基于DM的AP提供了对新攻击的有效适应能力。大量的实验表明，我们的方法达到了最先进的结果，并对不同的攻击表现出了泛化能力。



## **37. To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now**

生成还是不生成？安全驱动的未学习扩散模型仍然容易生成不安全的图像现在 cs.CV

Codes are available at  https://github.com/OPTML-Group/Diffusion-MU-Attack

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2310.11868v2) [paper-pdf](http://arxiv.org/pdf/2310.11868v2)

**Authors**: Yimeng Zhang, Jinghan Jia, Xin Chen, Aochuan Chen, Yihua Zhang, Jiancheng Liu, Ke Ding, Sijia Liu

**Abstract**: The recent advances in diffusion models (DMs) have revolutionized the generation of realistic and complex images. However, these models also introduce potential safety hazards, such as producing harmful content and infringing data copyrights. Despite the development of safety-driven unlearning techniques to counteract these challenges, doubts about their efficacy persist. To tackle this issue, we introduce an evaluation framework that leverages adversarial prompts to discern the trustworthiness of these safety-driven DMs after they have undergone the process of unlearning harmful concepts. Specifically, we investigated the adversarial robustness of DMs, assessed by adversarial prompts, when eliminating unwanted concepts, styles, and objects. We develop an effective and efficient adversarial prompt generation approach for DMs, termed UnlearnDiffAtk. This method capitalizes on the intrinsic classification abilities of DMs to simplify the creation of adversarial prompts, thereby eliminating the need for auxiliary classification or diffusion models.Through extensive benchmarking, we evaluate the robustness of five widely-used safety-driven unlearned DMs (i.e., DMs after unlearning undesirable concepts, styles, or objects) across a variety of tasks. Our results demonstrate the effectiveness and efficiency merits of UnlearnDiffAtk over the state-of-the-art adversarial prompt generation method and reveal the lack of robustness of current safety-driven unlearning techniques when applied to DMs. Codes are available at https://github.com/OPTML-Group/Diffusion-MU-Attack. WARNING: This paper contains model outputs that may be offensive in nature.

摘要: 扩散模型的最新进展使逼真和复杂图像的生成发生了革命性的变化。然而，这些模式也带来了潜在的安全隐患，如产生有害内容和侵犯数据著作权。尽管发展了安全驱动的遗忘技术来应对这些挑战，但对其有效性的怀疑依然存在。为了解决这个问题，我们引入了一个评估框架，利用对抗性提示，在这些以安全为导向的DM经历了忘记有害概念的过程后，识别他们的可信度。具体地说，我们研究了DM在消除不需要的概念、风格和对象时，通过对抗性提示评估的对抗性健壮性。本文提出了一种高效的敌意提示生成方法，称为UnlearnDiffAtk。该方法利用DM固有的分类能力来简化敌意提示的生成，从而消除了对辅助分类或扩散模型的需要。通过广泛的基准测试，我们评估了五种广泛使用的安全驱动的未学习DM(即忘记不良概念、风格或对象后的DM)在不同任务中的健壮性。实验结果证明了UnlearnDiffAtk算法相对于最新的对抗性提示生成方法的有效性和高效性，并揭示了当前安全驱动的遗忘技术在应用于决策支持系统时的健壮性不足。有关代码，请访问https://github.com/OPTML-Group/Diffusion-MU-Attack.警告：本文包含可能具有攻击性的模型输出。



## **38. An Embarrassingly Simple Defense Against Backdoor Attacks On SSL**

一个令人尴尬的简单的后门攻击防御SSL cs.CV

10 pages, 5 figures

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2403.15918v1) [paper-pdf](http://arxiv.org/pdf/2403.15918v1)

**Authors**: Aryan Satpathy, Nilaksh, Dhruva Rajwade

**Abstract**: Self Supervised Learning (SSL) has emerged as a powerful paradigm to tackle data landscapes with absence of human supervision. The ability to learn meaningful tasks without the use of labeled data makes SSL a popular method to manage large chunks of data in the absence of labels. However, recent work indicates SSL to be vulnerable to backdoor attacks, wherein models can be controlled, possibly maliciously, to suit an adversary's motives. Li et.al (2022) introduce a novel frequency-based backdoor attack: CTRL. They show that CTRL can be used to efficiently and stealthily gain control over a victim's model trained using SSL. In this work, we devise two defense strategies against frequency-based attacks in SSL: One applicable before model training and the second to be applied during model inference. Our first contribution utilizes the invariance property of the downstream task to defend against backdoor attacks in a generalizable fashion. We observe the ASR (Attack Success Rate) to reduce by over 60% across experiments. Our Inference-time defense relies on evasiveness of the attack and uses the luminance channel to defend against attacks. Using object classification as the downstream task for SSL, we demonstrate successful defense strategies that do not require re-training of the model. Code is available at https://github.com/Aryan-Satpathy/Backdoor.

摘要: 自我监督学习(SSL)已经成为一种强大的范式，可以在缺乏人类监督的情况下处理数据环境。无需使用标签数据即可学习有意义的任务的能力使SSL成为在没有标签的情况下管理大量数据的流行方法。然而，最近的研究表明，SSL容易受到后门攻击，在后门攻击中，可以控制模型，可能是恶意的，以适应对手的动机。Li et.al(2022)引入了一种新的基于频率的后门攻击：Ctrl。他们表明，CTRL可以用来有效地、秘密地控制使用SSL训练的受害者模型。在这项工作中，我们针对基于频率的攻击设计了两种防御策略：一种适用于模型训练之前，另一种应用于模型推理中。我们的第一个贡献是利用下游任务的不变性以一种可推广的方式防御后门攻击。我们观察到，在整个实验中，ASR(攻击成功率)降低了60%以上。我们的推理时间防御依赖于攻击的规避，并使用亮度通道来防御攻击。使用对象分类作为SSL的下游任务，我们演示了成功的防御策略，不需要对模型进行重新训练。代码可在https://github.com/Aryan-Satpathy/Backdoor.上找到



## **39. Effect of Ambient-Intrinsic Dimension Gap on Adversarial Vulnerability**

环境—内在维度差异对对抗脆弱性的影响 cs.LG

AISTATS 2024

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2403.03967v2) [paper-pdf](http://arxiv.org/pdf/2403.03967v2)

**Authors**: Rajdeep Haldar, Yue Xing, Qifan Song

**Abstract**: The existence of adversarial attacks on machine learning models imperceptible to a human is still quite a mystery from a theoretical perspective. In this work, we introduce two notions of adversarial attacks: natural or on-manifold attacks, which are perceptible by a human/oracle, and unnatural or off-manifold attacks, which are not. We argue that the existence of the off-manifold attacks is a natural consequence of the dimension gap between the intrinsic and ambient dimensions of the data. For 2-layer ReLU networks, we prove that even though the dimension gap does not affect generalization performance on samples drawn from the observed data space, it makes the clean-trained model more vulnerable to adversarial perturbations in the off-manifold direction of the data space. Our main results provide an explicit relationship between the $\ell_2,\ell_{\infty}$ attack strength of the on/off-manifold attack and the dimension gap.

摘要: 从理论角度来看，对机器学习模型的对抗性攻击的存在仍然是一个谜。在这项工作中，我们引入了两个概念的对抗攻击：自然或流形上攻击，这是人类/甲骨文可感知的，和非自然或非流形攻击，这是不。我们认为，非流形攻击的存在是数据的内在维度和环境维度之间的维度差距的自然结果。对于2层ReLU网络，我们证明了即使维数差距不影响从观测数据空间提取的样本的泛化性能，它使干净训练的模型更容易受到数据空间的非流形方向上的对抗扰动。我们的主要结果提供了$\ell_2，\ell_{\infty}$攻击强度与维数差距之间的明确关系。



## **40. Adversarial Defense Teacher for Cross-Domain Object Detection under Poor Visibility Conditions**

弱可见性条件下跨域目标检测的对抗性防御教师 cs.CV

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2403.15786v1) [paper-pdf](http://arxiv.org/pdf/2403.15786v1)

**Authors**: Kaiwen Wang, Yinzhe Shen, Martin Lauer

**Abstract**: Existing object detectors encounter challenges in handling domain shifts between training and real-world data, particularly under poor visibility conditions like fog and night. Cutting-edge cross-domain object detection methods use teacher-student frameworks and compel teacher and student models to produce consistent predictions under weak and strong augmentations, respectively. In this paper, we reveal that manually crafted augmentations are insufficient for optimal teaching and present a simple yet effective framework named Adversarial Defense Teacher (ADT), leveraging adversarial defense to enhance teaching quality. Specifically, we employ adversarial attacks, encouraging the model to generalize on subtly perturbed inputs that effectively deceive the model. To address small objects under poor visibility conditions, we propose a Zoom-in Zoom-out strategy, which zooms-in images for better pseudo-labels and zooms-out images and pseudo-labels to learn refined features. Our results demonstrate that ADT achieves superior performance, reaching 54.5% mAP on Foggy Cityscapes, surpassing the previous state-of-the-art by 2.6% mAP.

摘要: 现有的目标检测器在处理训练数据和真实世界数据之间的域转换方面遇到了挑战，特别是在能见度较低的条件下，如雾和夜晚。前沿的跨域目标检测方法使用教师-学生框架，迫使教师和学生模型分别在弱增长和强增长下产生一致的预测。在本文中，我们揭示了手工制作的扩充不足以优化教学，并提出了一个简单而有效的框架--对抗性防御教师(ADT)，利用对抗性防御来提高教学质量。具体地说，我们使用对抗性攻击，鼓励模型对微妙的扰动输入进行泛化，从而有效地欺骗模型。针对可见度较低条件下的小目标，我们提出了一种放大和缩小策略，该策略放大图像以获得更好的伪标签，而缩小图像和伪标签来学习精细特征。实验结果表明，ADT在雾天城市景观上取得了较好的性能，达到了54.5%的MAP，超过了以往最先进的2.6%的MAP。



## **41. Breaking Down the Defenses: A Comparative Survey of Attacks on Large Language Models**

突破防御：大型语言模型攻击的比较研究 cs.CR

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2403.04786v2) [paper-pdf](http://arxiv.org/pdf/2403.04786v2)

**Authors**: Arijit Ghosh Chowdhury, Md Mofijul Islam, Vaibhav Kumar, Faysal Hossain Shezan, Vaibhav Kumar, Vinija Jain, Aman Chadha

**Abstract**: Large Language Models (LLMs) have become a cornerstone in the field of Natural Language Processing (NLP), offering transformative capabilities in understanding and generating human-like text. However, with their rising prominence, the security and vulnerability aspects of these models have garnered significant attention. This paper presents a comprehensive survey of the various forms of attacks targeting LLMs, discussing the nature and mechanisms of these attacks, their potential impacts, and current defense strategies. We delve into topics such as adversarial attacks that aim to manipulate model outputs, data poisoning that affects model training, and privacy concerns related to training data exploitation. The paper also explores the effectiveness of different attack methodologies, the resilience of LLMs against these attacks, and the implications for model integrity and user trust. By examining the latest research, we provide insights into the current landscape of LLM vulnerabilities and defense mechanisms. Our objective is to offer a nuanced understanding of LLM attacks, foster awareness within the AI community, and inspire robust solutions to mitigate these risks in future developments.

摘要: 大型语言模型(LLM)已经成为自然语言处理(NLP)领域的基石，在理解和生成类似人类的文本方面提供了变革性的能力。然而，随着它们的日益突出，这些模型的安全和漏洞方面已经引起了极大的关注。本文对各种形式的针对LLMS的攻击进行了全面的综述，讨论了这些攻击的性质和机制、它们的潜在影响以及当前的防御策略。我们深入探讨了旨在操纵模型输出的对抗性攻击、影响模型训练的数据中毒以及与训练数据利用相关的隐私问题等主题。文中还探讨了不同攻击方法的有效性，LLMS对这些攻击的恢复能力，以及对模型完整性和用户信任的影响。通过检查最新的研究，我们提供了对LLM漏洞和防御机制的当前情况的见解。我们的目标是提供对LLM攻击的细微差别的理解，培养人工智能社区的意识，并激发强大的解决方案，以减轻未来发展中的这些风险。



## **42. On the Privacy Effect of Data Enhancement via the Lens of Memorization**

从加密化的角度看数据增强的隐私效应 cs.LG

Accepted by IEEE TIFS, 17 pages

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2208.08270v4) [paper-pdf](http://arxiv.org/pdf/2208.08270v4)

**Authors**: Xiao Li, Qiongxiu Li, Zhanhao Hu, Xiaolin Hu

**Abstract**: Machine learning poses severe privacy concerns as it has been shown that the learned models can reveal sensitive information about their training data. Many works have investigated the effect of widely adopted data augmentation and adversarial training techniques, termed data enhancement in the paper, on the privacy leakage of machine learning models. Such privacy effects are often measured by membership inference attacks (MIAs), which aim to identify whether a particular example belongs to the training set or not. We propose to investigate privacy from a new perspective called memorization. Through the lens of memorization, we find that previously deployed MIAs produce misleading results as they are less likely to identify samples with higher privacy risks as members compared to samples with low privacy risks. To solve this problem, we deploy a recent attack that can capture individual samples' memorization degrees for evaluation. Through extensive experiments, we unveil several findings about the connections between three essential properties of machine learning models, including privacy, generalization gap, and adversarial robustness. We demonstrate that the generalization gap and privacy leakage are less correlated than those of the previous results. Moreover, there is not necessarily a trade-off between adversarial robustness and privacy as stronger adversarial robustness does not make the model more susceptible to privacy attacks.

摘要: 机器学习带来了严重的隐私问题，因为已经表明，学习的模型可能会泄露关于其训练数据的敏感信息。许多工作已经研究了广泛采用的数据增强和对抗性训练技术对机器学习模型隐私泄露的影响。这种隐私影响通常通过成员关系推理攻击(MIA)来衡量，其目的是识别特定示例是否属于训练集。我们建议从一种名为记忆的新角度来研究隐私。通过记忆的镜头，我们发现之前部署的MIA产生了误导性的结果，因为与低隐私风险的样本相比，它们不太可能识别出隐私风险较高的样本作为成员。为了解决这个问题，我们部署了一个最近的攻击，可以捕获单个样本的记忆程度来进行评估。通过广泛的实验，我们揭示了关于机器学习模型的三个基本属性之间的联系的几个发现，包括隐私、泛化差距和对手健壮性。我们证明，泛化差距与隐私泄露之间的相关性低于之前的结果。此外，在对抗稳健性和隐私之间不一定存在权衡，因为更强的对抗稳健性并不会使模型更容易受到隐私攻击。



## **43. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

利用潜在对抗训练防御不可预见的故障模式 cs.CR

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.05030v2) [paper-pdf](http://arxiv.org/pdf/2403.05030v2)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: AI systems sometimes exhibit harmful unintended behaviors post-deployment. This is often despite extensive diagnostics and debugging by developers. Minimizing risks from models is challenging because the attack surface is so large. It is not tractable to exhaustively search for inputs that may cause a model to fail. Red-teaming and adversarial training (AT) are commonly used to make AI systems more robust. However, they have not been sufficient to avoid many real-world failure modes that differ from the ones adversarially trained on. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without generating inputs that elicit them. LAT leverages the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. We use LAT to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.

摘要: 人工智能系统有时会在部署后表现出有害的意外行为。这通常是尽管开发人员进行了广泛的诊断和调试。将模型的风险降至最低是具有挑战性的，因为攻击面如此之大。要详尽地搜索可能导致模型失败的输入是不容易的。红队和对抗训练(AT)通常被用来使AI系统更健壮。然而，它们还不足以避免许多现实世界中的失败模式，这些模式与对手训练的模式不同。在这项工作中，我们利用潜在的对手训练(LAT)来防御漏洞，而不会生成引发漏洞的输入。随后，利用网络实际用于预测的概念的压缩、抽象和结构化的潜在表示。我们使用LAT来删除特洛伊木马程序，并防御抵抗类的对抗性攻击。我们在图像分类、文本分类和文本生成任务中表明，与AT相比，LAT通常可以提高对干净数据的稳健性和性能。这表明，LAT可以成为一种很有前途的工具，用于防御开发人员未明确识别的故障模式。



## **44. From Hardware Fingerprint to Access Token: Enhancing the Authentication on IoT Devices**

从硬件指纹到访问令牌：增强物联网设备的认证 cs.CR

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.15271v1) [paper-pdf](http://arxiv.org/pdf/2403.15271v1)

**Authors**: Yue Xiao, Yi He, Xiaoli Zhang, Qian Wang, Renjie Xie, Kun Sun, Ke Xu, Qi Li

**Abstract**: The proliferation of consumer IoT products in our daily lives has raised the need for secure device authentication and access control. Unfortunately, these resource-constrained devices typically use token-based authentication, which is vulnerable to token compromise attacks that allow attackers to impersonate the devices and perform malicious operations by stealing the access token. Using hardware fingerprints to secure their authentication is a promising way to mitigate these threats. However, once attackers have stolen some hardware fingerprints (e.g., via MitM attacks), they can bypass the hardware authentication by training a machine learning model to mimic fingerprints or reusing these fingerprints to craft forge requests.   In this paper, we present MCU-Token, a secure hardware fingerprinting framework for MCU-based IoT devices even if the cryptographic mechanisms (e.g., private keys) are compromised. MCU-Token can be easily integrated with various IoT devices by simply adding a short hardware fingerprint-based token to the existing payload. To prevent the reuse of this token, we propose a message mapping approach that binds the token to a specific request via generating the hardware fingerprints based on the request payload. To defeat the machine learning attacks, we mix the valid fingerprints with poisoning data so that attackers cannot train a usable model with the leaked tokens. MCU-Token can defend against armored adversary who may replay, craft, and offload the requests via MitM or use both hardware (e.g., use identical devices) and software (e.g., machine learning attacks) strategies to mimic the fingerprints. The system evaluation shows that MCU-Token can achieve high accuracy (over 97%) with a low overhead across various IoT devices and application scenarios.

摘要: 日常生活中消费者物联网产品的激增提高了对安全设备身份验证和访问控制的需求。遗憾的是，这些资源受限的设备通常使用基于令牌的身份验证，这容易受到令牌泄露攻击，从而允许攻击者冒充设备并通过窃取访问令牌来执行恶意操作。使用硬件指纹保护他们的身份验证是缓解这些威胁的一种有前途的方法。然而，一旦攻击者窃取了一些硬件指纹(例如，通过MITM攻击)，他们就可以通过训练机器学习模型来模仿指纹或重复使用这些指纹来伪造请求，从而绕过硬件身份验证。本文提出了一种基于MCU的物联网设备安全硬件指纹识别框架MCU-Token，该框架可以在密码机制(例如私钥)被攻破的情况下提供安全的硬件指纹识别。MCU-Token只需在现有有效负载中添加一个简短的基于硬件指纹的令牌，就可以轻松地与各种物联网设备集成。为了防止令牌被重复使用，我们提出了一种消息映射方法，通过根据请求负载生成硬件指纹来将令牌绑定到特定的请求。为了击败机器学习攻击，我们将有效指纹与中毒数据混合，这样攻击者就无法使用泄露的令牌来训练可用的模型。MCU-Token可以防御可能通过MITM重放、手工和卸载请求或使用硬件(例如，使用相同的设备)和软件(例如，机器学习攻击)策略来模仿指纹的装甲攻击者。系统评估表明，MCU-Token能够以较低的开销实现跨各种物联网设备和应用场景的高准确率(97%以上)。



## **45. Robust optimization for adversarial learning with finite sample complexity guarantees**

具有有限样本复杂度保证的对抗学习鲁棒优化 cs.LG

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.15207v1) [paper-pdf](http://arxiv.org/pdf/2403.15207v1)

**Authors**: André Bertolace, Konstatinos Gatsis, Kostas Margellos

**Abstract**: Decision making and learning in the presence of uncertainty has attracted significant attention in view of the increasing need to achieve robust and reliable operations. In the case where uncertainty stems from the presence of adversarial attacks this need is becoming more prominent. In this paper we focus on linear and nonlinear classification problems and propose a novel adversarial training method for robust classifiers, inspired by Support Vector Machine (SVM) margins. We view robustness under a data driven lens, and derive finite sample complexity bounds for both linear and non-linear classifiers in binary and multi-class scenarios. Notably, our bounds match natural classifiers' complexity. Our algorithm minimizes a worst-case surrogate loss using Linear Programming (LP) and Second Order Cone Programming (SOCP) for linear and non-linear models. Numerical experiments on the benchmark MNIST and CIFAR10 datasets show our approach's comparable performance to state-of-the-art methods, without needing adversarial examples during training. Our work offers a comprehensive framework for enhancing binary linear and non-linear classifier robustness, embedding robustness in learning under the presence of adversaries.

摘要: 鉴于日益需要实现稳健和可靠的业务，在不确定情况下的决策和学习引起了极大的关注。在不确定性源于对抗性攻击的情况下，这一需要正变得更加突出。本文针对线性和非线性分类问题，借鉴支持向量机(Support VectorMachine，简称：支持向量机)的边缘特征，提出了一种新的对抗性的稳健分类器训练方法。我们在数据驱动的镜头下观察了稳健性，并推导了二类和多类情况下线性和非线性分类器的有限样本复杂性界限。值得注意的是，我们的界限与自然分类器的复杂性相匹配。对于线性和非线性模型，我们的算法使用线性规划(LP)和二阶锥规划(SOCP)最小化最坏情况下的代理损失。在基准MNIST和CIFAR10数据集上的数值实验表明，该方法的性能与最新方法相当，在训练过程中不需要对抗性例子。我们的工作提供了一个全面的框架来增强二进制线性和非线性分类器的稳健性，在存在对手的情况下嵌入学习的稳健性。



## **46. Privacy-Preserving End-to-End Spoken Language Understanding**

隐私保护的端到端口语理解 cs.CR

Accepted by IJCAI

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.15510v1) [paper-pdf](http://arxiv.org/pdf/2403.15510v1)

**Authors**: Yinggui Wang, Wei Huang, Le Yang

**Abstract**: Spoken language understanding (SLU), one of the key enabling technologies for human-computer interaction in IoT devices, provides an easy-to-use user interface. Human speech can contain a lot of user-sensitive information, such as gender, identity, and sensitive content. New types of security and privacy breaches have thus emerged. Users do not want to expose their personal sensitive information to malicious attacks by untrusted third parties. Thus, the SLU system needs to ensure that a potential malicious attacker cannot deduce the sensitive attributes of the users, while it should avoid greatly compromising the SLU accuracy. To address the above challenge, this paper proposes a novel SLU multi-task privacy-preserving model to prevent both the speech recognition (ASR) and identity recognition (IR) attacks. The model uses the hidden layer separation technique so that SLU information is distributed only in a specific portion of the hidden layer, and the other two types of information are removed to obtain a privacy-secure hidden layer. In order to achieve good balance between efficiency and privacy, we introduce a new mechanism of model pre-training, namely joint adversarial training, to further enhance the user privacy. Experiments over two SLU datasets show that the proposed method can reduce the accuracy of both the ASR and IR attacks close to that of a random guess, while leaving the SLU performance largely unaffected.

摘要: 口语理解(SLU)是物联网设备中实现人机交互的关键技术之一，它提供了易于使用的用户界面。人类语音可能包含大量用户敏感信息，如性别、身份和敏感内容。因此，出现了新类型的安全和隐私违规行为。用户不想将他们的个人敏感信息暴露在不受信任的第三方的恶意攻击下。因此，SLU系统需要确保潜在的恶意攻击者不能推断用户的敏感属性，同时应该避免极大地损害SLU的准确性。针对上述挑战，提出了一种新的SLU多任务隐私保护模型，以同时防止语音识别(ASR)和身份识别(IR)攻击。该模型使用隐层分离技术，使得SLU信息只分布在隐层的特定部分，而其他两种类型的信息被移除以获得隐私安全的隐层。为了在效率和隐私之间取得良好的平衡，我们引入了一种新的模型预训练机制，即联合对抗性训练，以进一步增强用户的隐私。在两个SLU数据集上的实验表明，该方法可以将ASR和IR攻击的准确率降低到接近随机猜测的水平，而SLU的性能基本不受影响。



## **47. TTPXHunter: Actionable Threat Intelligence Extraction as TTPs from Finished Cyber Threat Reports**

TTPXHunter：从完成的网络威胁报告中提取可操作的威胁情报 cs.CR

Under Review

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.03267v3) [paper-pdf](http://arxiv.org/pdf/2403.03267v3)

**Authors**: Nanda Rani, Bikash Saha, Vikas Maurya, Sandeep Kumar Shukla

**Abstract**: Understanding the modus operandi of adversaries aids organizations in employing efficient defensive strategies and sharing intelligence in the community. This knowledge is often present in unstructured natural language text within threat analysis reports. A translation tool is needed to interpret the modus operandi explained in the sentences of the threat report and translate it into a structured format. This research introduces a methodology named TTPXHunter for the automated extraction of threat intelligence in terms of Tactics, Techniques, and Procedures (TTPs) from finished cyber threat reports. It leverages cyber domain-specific state-of-the-art natural language processing (NLP) to augment sentences for minority class TTPs and refine pinpointing the TTPs in threat analysis reports significantly. The knowledge of threat intelligence in terms of TTPs is essential for comprehensively understanding cyber threats and enhancing detection and mitigation strategies. We create two datasets: an augmented sentence-TTP dataset of 39,296 samples and a 149 real-world cyber threat intelligence report-to-TTP dataset. Further, we evaluate TTPXHunter on the augmented sentence dataset and the cyber threat reports. The TTPXHunter achieves the highest performance of 92.42% f1-score on the augmented dataset, and it also outperforms existing state-of-the-art solutions in TTP extraction by achieving an f1-score of 97.09% when evaluated over the report dataset. TTPXHunter significantly improves cybersecurity threat intelligence by offering quick, actionable insights into attacker behaviors. This advancement automates threat intelligence analysis, providing a crucial tool for cybersecurity professionals fighting cyber threats.

摘要: 了解对手的作案手法有助于组织采用有效的防御策略，并在社区中分享情报。这种知识通常出现在威胁分析报告中的非结构化自然语言文本中。需要一个翻译工具来解释威胁报告句子中解释的工作方式，并将其翻译成结构化格式。本研究介绍了一种名为TTPXHunter的方法，用于从已完成的网络威胁报告中自动提取策略、技术和过程(TTP)方面的威胁情报。它利用特定于网络领域的最先进的自然语言处理(NLP)来增加少数族裔类TTP的句子，并显著细化威胁分析报告中的TTP。TTP方面的威胁情报知识对于全面了解网络威胁和加强检测和缓解战略至关重要。我们创建了两个数据集：一个包含39,296个样本的增强句-TTP数据集，以及149个真实世界网络威胁情报报告到TTP的数据集。此外，我们在增加句子数据集和网络威胁报告上对TTPXHunter进行了评估。TTPXHunter在增强的数据集上获得了92.42%的F1分数的最高性能，在TTP提取方面也超过了现有的最先进的解决方案，在报告数据集上的F1分数达到了97.09%。TTPXHunter通过提供对攻击者行为的快速、可操作的洞察，显著提高了网络安全威胁情报。这一进步使威胁情报分析自动化，为应对网络威胁的网络安全专业人员提供了一个重要工具。



## **48. Diffusion Attack: Leveraging Stable Diffusion for Naturalistic Image Attacking**

扩散攻击：利用稳定扩散进行自然图像攻击 cs.CV

Accepted to IEEE VRW

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14778v1) [paper-pdf](http://arxiv.org/pdf/2403.14778v1)

**Authors**: Qianyu Guo, Jiaming Fu, Yawen Lu, Dongming Gan

**Abstract**: In Virtual Reality (VR), adversarial attack remains a significant security threat. Most deep learning-based methods for physical and digital adversarial attacks focus on enhancing attack performance by crafting adversarial examples that contain large printable distortions that are easy for human observers to identify. However, attackers rarely impose limitations on the naturalness and comfort of the appearance of the generated attack image, resulting in a noticeable and unnatural attack. To address this challenge, we propose a framework to incorporate style transfer to craft adversarial inputs of natural styles that exhibit minimal detectability and maximum natural appearance, while maintaining superior attack capabilities.

摘要: 在虚拟现实（VR）中，对抗性攻击仍然是一个重要的安全威胁。大多数基于深度学习的物理和数字对抗攻击方法都专注于通过制作包含大量可打印失真的对抗示例来增强攻击性能，这些失真易于人类观察者识别。然而，攻击者很少对生成的攻击图像外观的自然性和舒适性施加限制，导致明显和不自然的攻击。为了解决这一挑战，我们提出了一个框架，将风格转移纳入工艺对抗输入的自然风格，表现出最小的可检测性和最大的自然外观，同时保持卓越的攻击能力。



## **49. Improving Robustness to Model Inversion Attacks via Sparse Coding Architectures**

基于稀疏编码结构的反向攻击建模鲁棒性提高 cs.CV

32 pages, 15 Tables, and 9 Figures

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14772v1) [paper-pdf](http://arxiv.org/pdf/2403.14772v1)

**Authors**: Sayanton V. Dibbo, Adam Breuer, Juston Moore, Michael Teti

**Abstract**: Recent model inversion attack algorithms permit adversaries to reconstruct a neural network's private training data just by repeatedly querying the network and inspecting its outputs. In this work, we develop a novel network architecture that leverages sparse-coding layers to obtain superior robustness to this class of attacks. Three decades of computer science research has studied sparse coding in the context of image denoising, object recognition, and adversarial misclassification settings, but to the best of our knowledge, its connection to state-of-the-art privacy vulnerabilities remains unstudied. However, sparse coding architectures suggest an advantageous means to defend against model inversion attacks because they allow us to control the amount of irrelevant private information encoded in a network's intermediate representations in a manner that can be computed efficiently during training and that is known to have little effect on classification accuracy. Specifically, compared to networks trained with a variety of state-of-the-art defenses, our sparse-coding architectures maintain comparable or higher classification accuracy while degrading state-of-the-art training data reconstructions by factors of 1.1 to 18.3 across a variety of reconstruction quality metrics (PSNR, SSIM, FID). This performance advantage holds across 5 datasets ranging from CelebA faces to medical images and CIFAR-10, and across various state-of-the-art SGD-based and GAN-based inversion attacks, including Plug-&-Play attacks. We provide a cluster-ready PyTorch codebase to promote research and standardize defense evaluations.

摘要: 最近的模型反转攻击算法允许攻击者只需重复查询网络并检查其输出即可重建神经网络的私有训练数据。在这项工作中，我们开发了一种新颖的网络体系结构，该体系结构利用稀疏编码层来获得对此类攻击的卓越健壮性。三十年的计算机科学研究已经在图像去噪、目标识别和敌意错误分类环境中研究了稀疏编码，但就我们所知，它与最先进的隐私漏洞的联系仍未被研究。然而，稀疏编码体系结构建议了一种防御模型反转攻击的有利手段，因为它们允许我们控制编码在网络的中间表示中的无关私有信息量，这种方式可以在训练期间有效地计算，并且已知对分类精度几乎没有影响。具体地说，与使用各种最先进的防御措施训练的网络相比，我们的稀疏编码体系结构保持了相当或更高的分类精度，同时在各种重建质量指标(PSNR、SSIM、FID)上将最先进的训练数据重建降级1.1至18.3倍。这一性能优势涵盖从CelebA Faces到医学图像和CIFAR-10的5个数据集，以及各种最先进的基于SGD和GAN的反转攻击，包括即插即用攻击。我们提供了一个支持集群的PyTorch代码库，以促进研究和标准化防御评估。



## **50. TMI! Finetuned Models Leak Private Information from their Pretraining Data**

TMI！精细调谐模型从预训练数据中泄漏私人信息 cs.LG

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2306.01181v2) [paper-pdf](http://arxiv.org/pdf/2306.01181v2)

**Authors**: John Abascal, Stanley Wu, Alina Oprea, Jonathan Ullman

**Abstract**: Transfer learning has become an increasingly popular technique in machine learning as a way to leverage a pretrained model trained for one task to assist with building a finetuned model for a related task. This paradigm has been especially popular for $\textit{privacy}$ in machine learning, where the pretrained model is considered public, and only the data for finetuning is considered sensitive. However, there are reasons to believe that the data used for pretraining is still sensitive, making it essential to understand how much information the finetuned model leaks about the pretraining data. In this work we propose a new membership-inference threat model where the adversary only has access to the finetuned model and would like to infer the membership of the pretraining data. To realize this threat model, we implement a novel metaclassifier-based attack, $\textbf{TMI}$, that leverages the influence of memorized pretraining samples on predictions in the downstream task. We evaluate $\textbf{TMI}$ on both vision and natural language tasks across multiple transfer learning settings, including finetuning with differential privacy. Through our evaluation, we find that $\textbf{TMI}$ can successfully infer membership of pretraining examples using query access to the finetuned model. An open-source implementation of $\textbf{TMI}$ can be found $\href{https://github.com/johnmath/tmi-pets24}{\text{on GitHub}}$.

摘要: 迁移学习已经成为机器学习中一种越来越流行的技术，作为一种利用为一个任务训练的预先训练的模型来帮助构建相关任务的精调模型的一种方式。这种范例在机器学习中尤其流行，在机器学习中，预先训练的模型被认为是公共的，只有用于精细调整的数据被认为是敏感的。然而，有理由相信，用于预培训的数据仍然敏感，因此了解精调模型泄露了多少关于预培训数据的信息是至关重要的。在这项工作中，我们提出了一个新的成员资格推理威胁模型，其中对手只能访问精调的模型，并希望推断预训练数据的成员资格。为了实现这种威胁模型，我们实现了一种新的基于元分类器的攻击，它利用已记忆的预训练样本对下游任务预测的影响。我们在多个迁移学习环境中对视觉和自然语言任务进行了评估，包括在不同隐私条件下的精细调整。通过我们的评估，我们发现$\textbf{TMI}$可以成功地使用对精调模型的查询访问来推断预训练样本的隶属度。$\Textbf{tmi}$的开源实现可以在$\href{https://github.com/johnmath/tmi-pets24}{\text{on GitHub中找到。



