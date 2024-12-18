# Latest Large Language Model Attack Papers
**update at 2024-12-18 10:29:20**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Truthful Text Sanitization Guided by Inference Attacks**

推理攻击引导的真实文本清理 cs.CL

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12928v1) [paper-pdf](http://arxiv.org/pdf/2412.12928v1)

**Authors**: Ildikó Pilán, Benet Manzanares-Salor, David Sánchez, Pierre Lison

**Abstract**: The purpose of text sanitization is to rewrite those text spans in a document that may directly or indirectly identify an individual, to ensure they no longer disclose personal information. Text sanitization must strike a balance between preventing the leakage of personal information (privacy protection) while also retaining as much of the document's original content as possible (utility preservation). We present an automated text sanitization strategy based on generalizations, which are more abstract (but still informative) terms that subsume the semantic content of the original text spans. The approach relies on instruction-tuned large language models (LLMs) and is divided into two stages. The LLM is first applied to obtain truth-preserving replacement candidates and rank them according to their abstraction level. Those candidates are then evaluated for their ability to protect privacy by conducting inference attacks with the LLM. Finally, the system selects the most informative replacement shown to be resistant to those attacks. As a consequence of this two-stage process, the chosen replacements effectively balance utility and privacy. We also present novel metrics to automatically evaluate these two aspects without the need to manually annotate data. Empirical results on the Text Anonymization Benchmark show that the proposed approach leads to enhanced utility, with only a marginal increase in the risk of re-identifying protected individuals compared to fully suppressing the original information. Furthermore, the selected replacements are shown to be more truth-preserving and abstractive than previous methods.

摘要: 文本清理的目的是重写文档中可能直接或间接识别个人身份的文本范围，以确保他们不再泄露个人信息。文本清理必须在防止个人信息泄露(隐私保护)和尽可能多地保留文档的原始内容(实用程序保护)之间取得平衡。我们提出了一种基于泛化的自动文本净化策略，泛化是更抽象(但仍然信息)的术语，包含了原始文本范围的语义内容。该方法依赖于指令调优的大型语言模型(LLM)，分为两个阶段。LLM首先用于获取保持真值的替换候选，并根据它们的抽象级别对它们进行排序。然后，对这些候选人进行评估，以确定他们通过使用LLM进行推理攻击来保护隐私的能力。最后，系统会选择能够抵抗这些攻击的信息最丰富的替代方案。作为这两个阶段过程的结果，所选择的替代品有效地平衡了实用性和隐私。我们还提出了新的度量标准来自动评估这两个方面，而不需要手动注释数据。在文本匿名化基准上的实验结果表明，与完全抑制原始信息相比，该方法提高了效用，而重新识别受保护个人的风险仅略有增加。此外，与以前的方法相比，所选择的替换方法具有更强的真实性和抽象性。



## **2. PROSAC: Provably Safe Certification for Machine Learning Models under Adversarial Attacks**

PROSAC：对抗性攻击下的机器学习模型可证明安全的认证 cs.LG

Accepted to AAAI2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2402.02629v2) [paper-pdf](http://arxiv.org/pdf/2402.02629v2)

**Authors**: Chen Feng, Ziquan Liu, Zhuo Zhi, Ilija Bogunovic, Carsten Gerner-Beuerle, Miguel Rodrigues

**Abstract**: It is widely known that state-of-the-art machine learning models, including vision and language models, can be seriously compromised by adversarial perturbations. It is therefore increasingly relevant to develop capabilities to certify their performance in the presence of the most effective adversarial attacks. Our paper offers a new approach to certify the performance of machine learning models in the presence of adversarial attacks with population level risk guarantees. In particular, we introduce the notion of $(\alpha,\zeta)$-safe machine learning model. We propose a hypothesis testing procedure, based on the availability of a calibration set, to derive statistical guarantees providing that the probability of declaring that the adversarial (population) risk of a machine learning model is less than $\alpha$ (i.e. the model is safe), while the model is in fact unsafe (i.e. the model adversarial population risk is higher than $\alpha$), is less than $\zeta$. We also propose Bayesian optimization algorithms to determine efficiently whether a machine learning model is $(\alpha,\zeta)$-safe in the presence of an adversarial attack, along with statistical guarantees. We apply our framework to a range of machine learning models - including various sizes of vision Transformer (ViT) and ResNet models - impaired by a variety of adversarial attacks, such as PGDAttack, MomentumAttack, GenAttack and BanditAttack, to illustrate the operation of our approach. Importantly, we show that ViT's are generally more robust to adversarial attacks than ResNets, and large models are generally more robust than smaller models. Our approach goes beyond existing empirical adversarial risk-based certification guarantees. It formulates rigorous (and provable) performance guarantees that can be used to satisfy regulatory requirements mandating the use of state-of-the-art technical tools.

摘要: 众所周知，最先进的机器学习模型，包括视觉和语言模型，可能会受到对抗性扰动的严重影响。因此，越来越有必要发展能力，以证明它们在最有效的对抗性攻击下的表现。本文提供了一种新的方法来证明机器学习模型在种群水平风险保证的对抗性攻击下的性能。特别地，我们引入了$(\α，\Zeta)$-安全机器学习模型的概念。我们提出了一种假设检验程序，基于校准集的可用性来获得统计保证，假设宣布一个机器学习模型的对抗(总体)风险小于$\α$(即该模型是安全的)，而该模型实际上是不安全的(即该模型的对抗总体风险高于$\α$)的概率小于$\Zeta$。我们还提出了贝叶斯优化算法来有效地确定机器学习模型在存在对抗性攻击的情况下是否$(\α，\Zeta)$安全，并提供统计保证。我们将我们的框架应用于一系列机器学习模型，包括各种大小的视觉转换器(VIT)和ResNet模型，这些模型被各种敌意攻击所破坏，如PGDAttack、MomentumAttack、GenAttack和BanditAttack，以说明我们方法的操作。重要的是，我们发现VIT通常比ResNet对对手攻击更健壮，大模型通常比小模型更健壮。我们的方法超越了现有的经验对抗性、基于风险的认证保证。它制定了严格的(和可证明的)性能保证，可用于满足要求使用最先进技术工具的监管要求。



## **3. RemoteRAG: A Privacy-Preserving LLM Cloud RAG Service**

远程RAG：保护隐私的LLM云RAG服务 cs.IR

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12775v1) [paper-pdf](http://arxiv.org/pdf/2412.12775v1)

**Authors**: Yihang Cheng, Lan Zhang, Junyang Wang, Mu Yuan, Yunhao Yao

**Abstract**: Retrieval-augmented generation (RAG) improves the service quality of large language models by retrieving relevant documents from credible literature and integrating them into the context of the user query. Recently, the rise of the cloud RAG service has made it possible for users to query relevant documents conveniently. However, directly sending queries to the cloud brings potential privacy leakage. In this paper, we are the first to formally define the privacy-preserving cloud RAG service to protect the user query and propose RemoteRAG as a solution regarding privacy, efficiency, and accuracy. For privacy, we introduce $(n,\epsilon)$-DistanceDP to characterize privacy leakage of the user query and the leakage inferred from relevant documents. For efficiency, we limit the search range from the total documents to a small number of selected documents related to a perturbed embedding generated from $(n,\epsilon)$-DistanceDP, so that computation and communication costs required for privacy protection significantly decrease. For accuracy, we ensure that the small range includes target documents related to the user query with detailed theoretical analysis. Experimental results also demonstrate that RemoteRAG can resist existing embedding inversion attack methods while achieving no loss in retrieval under various settings. Moreover, RemoteRAG is efficient, incurring only $0.67$ seconds and $46.66$KB of data transmission ($2.72$ hours and $1.43$ GB with the non-optimized privacy-preserving scheme) when retrieving from a total of $10^6$ documents.

摘要: 检索增强生成(RAG)通过从可信文献中检索相关文档并将其集成到用户查询的上下文中来提高大型语言模型的服务质量。最近，云抹布服务的兴起，使用户可以方便地查询相关文档。然而，直接向云端发送查询会带来潜在的隐私泄露。在本文中，我们首次正式定义了保护隐私的云RAG服务来保护用户查询，并提出了RemoteRAG作为一种隐私、效率和准确性的解决方案。在隐私方面，我们引入了$(n，\epsilon)$-DistanceDP来刻画用户查询的隐私泄露和从相关文档中推断的隐私泄露。为了提高搜索效率，我们将搜索范围从全部文档限制到与由$(n，\epsilon)$-DistanceDP生成的扰动嵌入相关的少量选定文档，从而显著降低了隐私保护所需的计算和通信代价。为了准确，我们确保小范围包括与用户查询相关的目标文档，并进行详细的理论分析。实验结果还表明，RemoteRAG可以抵抗现有的嵌入反转攻击方法，同时在不同的设置下实现了无损失的检索。此外，RemoteRAG是高效的，当从总共$10^6$文档中检索时，仅需花费$0.67$秒和$46.66$KB的数据传输(使用未经优化的隐私保护方案，$2.72$小时和$1.43$GB)。



## **4. Defending LVLMs Against Vision Attacks through Partial-Perception Supervision**

通过部分感知监督保护LVLM免受视觉攻击 cs.CV

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12722v1) [paper-pdf](http://arxiv.org/pdf/2412.12722v1)

**Authors**: Qi Zhou, Tianlin Li, Qing Guo, Dongxia Wang, Yun Lin, Yang Liu, Jin Song Dong

**Abstract**: Recent studies have raised significant concerns regarding the vulnerability of Large Vision Language Models (LVLMs) to maliciously injected or perturbed input images, which can mislead their responses. Existing defense methods show that such vision attacks are sensitive to image modifications especially cropping, using majority voting across responses of modified images as corrected responses. However, these modifications often result in partial images and distort the semantics, which reduces response quality on clean images after voting. Instead of directly using responses from partial images for voting, we investigate using them to supervise the LVLM's responses to the original images. We propose a black-box, training-free method called DPS (Defense through Partial-Perception Supervision). In this approach, the model is prompted using the responses generated by a model that perceives only a partial image. With DPS, the model can adjust its response based on partial image understanding when under attack, while confidently maintaining its original response for clean input. Our findings show that the weak model can supervise the strong model: when faced with an attacked input, the strong model becomes less confident and adjusts its response based on the weak model's partial understanding, effectively defending against the attack. With clean input, it confidently maintains its original response. Empirical experiments show our method outperforms the baseline, cutting the average attack success rate by 76.3% across six datasets on three popular models.

摘要: 最近的研究引起了人们对大视觉语言模型(LVMs)对恶意注入或干扰输入图像的脆弱性的严重关注，这可能会误导它们的反应。现有的防御方法表明，这种视觉攻击对图像修改特别是裁剪敏感，使用修改后的图像的响应中的多数投票作为校正响应。然而，这些修改往往导致图像不完整，扭曲了语义，降低了投票后对干净图像的响应质量。我们不是直接使用部分图像的响应进行投票，而是研究使用它们来监督LVLM对原始图像的响应。我们提出了一种黑盒、无需训练的方法，称为DPS(部分感知监控防御)。在该方法中，使用由仅感知部分图像的模型生成的响应来提示模型。有了DPS，该模型可以在受到攻击时基于对部分图像的理解来调整其响应，同时自信地保持其原始响应以进行干净的输入。我们的研究结果表明，弱模型可以监督强模型：当面对攻击输入时，强模型变得不那么自信，并根据弱模型的部分理解来调整其响应，从而有效地防御攻击。有了干净的输入，它自信地保持了原来的反应。实验结果表明，该方法在三种流行模型的六个数据集上的平均攻击成功率降低了76.3%。



## **5. Jailbreaking? One Step Is Enough!**

越狱？一步就够了！ cs.CL

17 pages

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12621v1) [paper-pdf](http://arxiv.org/pdf/2412.12621v1)

**Authors**: Weixiong Zheng, Peijian Zeng, Yiwei Li, Hongyan Wu, Nankai Lin, Junhao Chen, Aimin Yang, Yongmei Zhou

**Abstract**: Large language models (LLMs) excel in various tasks but remain vulnerable to jailbreak attacks, where adversaries manipulate prompts to generate harmful outputs. Examining jailbreak prompts helps uncover the shortcomings of LLMs. However, current jailbreak methods and the target model's defenses are engaged in an independent and adversarial process, resulting in the need for frequent attack iterations and redesigning attacks for different models. To address these gaps, we propose a Reverse Embedded Defense Attack (REDA) mechanism that disguises the attack intention as the "defense". intention against harmful content. Specifically, REDA starts from the target response, guiding the model to embed harmful content within its defensive measures, thereby relegating harmful content to a secondary role and making the model believe it is performing a defensive task. The attacking model considers that it is guiding the target model to deal with harmful content, while the target model thinks it is performing a defensive task, creating an illusion of cooperation between the two. Additionally, to enhance the model's confidence and guidance in "defensive" intentions, we adopt in-context learning (ICL) with a small number of attack examples and construct a corresponding dataset of attack examples. Extensive evaluations demonstrate that the REDA method enables cross-model attacks without the need to redesign attack strategies for different models, enables successful jailbreak in one iteration, and outperforms existing methods on both open-source and closed-source models.

摘要: 大型语言模型(LLM)在各种任务中表现出色，但仍然容易受到越狱攻击的攻击，在越狱攻击中，对手操纵提示以生成有害的输出。检查越狱提示有助于发现LLMS的缺点。然而，当前的越狱方法和目标模型的防御都是独立的和对抗性的过程，导致需要频繁的攻击迭代和针对不同模型的重新设计攻击。针对这些漏洞，我们提出了一种反向嵌入防御攻击(REDA)机制，将攻击意图伪装成“防御”。针对有害内容的意图。具体地说，Reda从目标响应开始，引导模型在其防御措施中嵌入有害内容，从而将有害内容降级为次要角色，并使模型相信它正在执行防御任务。攻击模型认为它是在引导目标模型处理有害内容，而目标模型则认为它是在执行防御任务，制造了两者合作的错觉。此外，为了增强模型对“防御”意图的可信度和指导性，我们采用了上下文中学习(ICL)的方法，并结合少量攻击实例构建了相应的攻击实例数据集。广泛的评估表明，REDA方法支持跨模型攻击，不需要针对不同的模型重新设计攻击策略，一次迭代即可成功越狱，并且在开源和闭源模型上的性能都优于现有方法。



## **6. Jailbreak Large Vision-Language Models Through Multi-Modal Linkage**

通过多模式联动的越狱大型视觉语言模型 cs.CV

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.00473v4) [paper-pdf](http://arxiv.org/pdf/2412.00473v4)

**Authors**: Yu Wang, Xiaofei Zhou, Yichen Wang, Geyuan Zhang, Tianxing He

**Abstract**: With the significant advancement of Large Vision-Language Models (VLMs), concerns about their potential misuse and abuse have grown rapidly. Previous studies have highlighted VLMs' vulnerability to jailbreak attacks, where carefully crafted inputs can lead the model to produce content that violates ethical and legal standards. However, existing methods struggle against state-of-the-art VLMs like GPT-4o, due to the over-exposure of harmful content and lack of stealthy malicious guidance. In this work, we propose a novel jailbreak attack framework: Multi-Modal Linkage (MML) Attack. Drawing inspiration from cryptography, MML utilizes an encryption-decryption process across text and image modalities to mitigate over-exposure of malicious information. To align the model's output with malicious intent covertly, MML employs a technique called "evil alignment", framing the attack within a video game production scenario. Comprehensive experiments demonstrate MML's effectiveness. Specifically, MML jailbreaks GPT-4o with attack success rates of 97.80% on SafeBench, 98.81% on MM-SafeBench and 99.07% on HADES-Dataset. Our code is available at https://github.com/wangyu-ovo/MML

摘要: 随着大型视觉语言模型(VLM)的显著进步，人们对其潜在的滥用和滥用的担忧迅速增长。之前的研究已经强调了VLMS在越狱攻击中的脆弱性，在越狱攻击中，精心制作的输入可能导致该模型产生违反道德和法律标准的内容。然而，由于过度暴露有害内容和缺乏隐蔽的恶意指导，现有的方法难以对抗像GPT-40这样的最先进的VLM。在这项工作中，我们提出了一种新的越狱攻击框架：多模式联动攻击。MML从密码学中获得灵感，利用跨文本和图像通道的加密-解密过程来减少恶意信息的过度暴露。为了秘密地将模型的输出与恶意意图对齐，MML采用了一种称为“邪恶对齐”的技术，将攻击框置于视频游戏制作场景中。综合实验证明了MML的有效性。具体地说，MML越狱GPT-4o在SafeBitch上的攻击成功率为97.80%，在MM-SafeBch上的攻击成功率为98.81%，在HADES-DataSet上的攻击成功率为99.07%。我们的代码可以在https://github.com/wangyu-ovo/MML上找到



## **7. Task-Agnostic Language Model Watermarking via High Entropy Passthrough Layers**

任务不可知语言模型通过高熵传递层进行水印 cs.CL

Accepted to AAAI2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12563v1) [paper-pdf](http://arxiv.org/pdf/2412.12563v1)

**Authors**: Vaden Masrani, Mohammad Akbari, David Ming Xuan Yue, Ahmad Rezaei, Yong Zhang

**Abstract**: In the era of costly pre-training of large language models, ensuring the intellectual property rights of model owners, and insuring that said models are responsibly deployed, is becoming increasingly important. To this end, we propose model watermarking via passthrough layers, which are added to existing pre-trained networks and trained using a self-supervised loss such that the model produces high-entropy output when prompted with a unique private key, and acts normally otherwise. Unlike existing model watermarking methods, our method is fully task-agnostic, and can be applied to both classification and sequence-to-sequence tasks without requiring advanced access to downstream fine-tuning datasets. We evaluate the proposed passthrough layers on a wide range of downstream tasks, and show experimentally our watermarking method achieves a near-perfect watermark extraction accuracy and false-positive rate in most cases without damaging original model performance. Additionally, we show our method is robust to both downstream fine-tuning, fine-pruning, and layer removal attacks, and can be trained in a fraction of the time required to train the original model. Code is available in the paper.

摘要: 在对大型语言模型进行昂贵的预培训的时代，确保模型所有者的知识产权并确保负责任地部署这些模型变得越来越重要。为此，我们提出了通过穿透层的模型水印，这些穿透层被添加到现有的预训练网络中，并使用自监督损失进行训练，使得模型在使用唯一私钥提示时产生高熵输出，否则正常工作。与现有的模型水印方法不同，我们的方法是完全与任务无关的，并且可以应用于分类和序列到序列的任务，而不需要提前访问下游微调数据集。实验表明，在不影响原始模型性能的前提下，我们的水印方法在大多数情况下都达到了近乎完美的水印提取精度和误检率。此外，我们还证明了我们的方法对下行微调、精细剪枝和层移除攻击都是健壮的，并且可以在训练原始模型所需时间的一小部分内进行训练。代码可以在报纸上找到。



## **8. Recent advancements in LLM Red-Teaming: Techniques, Defenses, and Ethical Considerations**

LLM红色团队的最新进展：技术、辩护和道德考虑 cs.CL

16 pages, 2 figures

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2410.09097v2) [paper-pdf](http://arxiv.org/pdf/2410.09097v2)

**Authors**: Tarun Raheja, Nilay Pochhi, F. D. C. M. Curie

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing tasks, but their vulnerability to jailbreak attacks poses significant security risks. This survey paper presents a comprehensive analysis of recent advancements in attack strategies and defense mechanisms within the field of Large Language Model (LLM) red-teaming. We analyze various attack methods, including gradient-based optimization, reinforcement learning, and prompt engineering approaches. We discuss the implications of these attacks on LLM safety and the need for improved defense mechanisms. This work aims to provide a thorough understanding of the current landscape of red-teaming attacks and defenses on LLMs, enabling the development of more secure and reliable language models.

摘要: 大型语言模型（LLM）在自然语言处理任务中表现出了非凡的能力，但它们对越狱攻击的脆弱性带来了巨大的安全风险。这篇调查论文全面分析了大型语言模型（LLM）红色团队领域攻击策略和防御机制的最新进展。我们分析了各种攻击方法，包括基于梯度的优化、强化学习和提示工程方法。我们讨论了这些攻击对LLM安全性的影响以及改进防御机制的必要性。这项工作旨在彻底了解LLC上红色团队攻击和防御的当前情况，从而开发更安全、更可靠的语言模型。



## **9. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

大型语言模型能否提高图神经网络的对抗鲁棒性？ cs.LG

accepted by KDD2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2408.08685v2) [paper-pdf](http://arxiv.org/pdf/2408.08685v2)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial attacks, especially for topology perturbations, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attacks. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph. The source code can be found in https://github.com/zhongjian-zhang/LLM4RGNN.

摘要: 图神经网络(GNN)容易受到敌意攻击，尤其是对拓扑扰动的攻击，许多提高GNN健壮性的方法受到了广泛的关注。最近，我们目睹了大型语言模型(LLM)的巨大成功，这导致许多人探索LLM在GNN上的巨大潜力。然而，它们主要集中在通过利用LLMS来增强节点特征来提高GNN的性能。因此，我们问：GNN的健壮性是否也会随着LLMS强大的理解和推理能力而得到增强？通过给出实验结果，我们发现，尽管LLMS可以提高GNN的健壮性，但其准确率仍然平均下降23.1%，这意味着GNN仍然非常容易受到拓扑攻击。因此，另一个问题是如何扩展LLMS在图对抗健壮性方面的能力。大量的实验表明，LLM4RGNN在不同的GNN上一致地提高了健壮性。即使在某些扰动比增加到40%的情况下，GNN的精度仍然好于干净图形上的精度。源代码可以在https://github.com/zhongjian-zhang/LLM4RGNN.中找到



## **10. NLSR: Neuron-Level Safety Realignment of Large Language Models Against Harmful Fine-Tuning**

NRSC：大型语言模型的神经元级安全重新调整以防止有害的微调 cs.CL

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12497v1) [paper-pdf](http://arxiv.org/pdf/2412.12497v1)

**Authors**: Xin Yi, Shunfan Zheng, Linlin Wang, Gerard de Melo, Xiaoling Wang, Liang He

**Abstract**: The emergence of finetuning-as-a-service has revealed a new vulnerability in large language models (LLMs). A mere handful of malicious data uploaded by users can subtly manipulate the finetuning process, resulting in an alignment-broken model. Existing methods to counteract fine-tuning attacks typically require substantial computational resources. Even with parameter-efficient techniques like LoRA, gradient updates remain essential. To address these challenges, we propose \textbf{N}euron-\textbf{L}evel \textbf{S}afety \textbf{R}ealignment (\textbf{NLSR}), a training-free framework that restores the safety of LLMs based on the similarity difference of safety-critical neurons before and after fine-tuning. The core of our framework is first to construct a safety reference model from an initially aligned model to amplify safety-related features in neurons. We then utilize this reference model to identify safety-critical neurons, which we prepare as patches. Finally, we selectively restore only those neurons that exhibit significant similarity differences by transplanting these prepared patches, thereby minimally altering the fine-tuned model. Extensive experiments demonstrate significant safety enhancements in fine-tuned models across multiple downstream tasks, while greatly maintaining task-level accuracy. Our findings suggest regions of some safety-critical neurons show noticeable differences after fine-tuning, which can be effectively corrected by transplanting neurons from the reference model without requiring additional training. The code will be available at \url{https://github.com/xinykou/NLSR}

摘要: 精调即服务的出现揭示了大型语言模型(LLM)中的一个新漏洞。用户上传的几个恶意数据就可以巧妙地操纵微调过程，导致对齐破坏模型。现有的对抗微调攻击的方法通常需要大量的计算资源。即使使用LORA这样的参数高效技术，渐变更新仍然是必不可少的。为了应对这些挑战，我们提出了一种无需训练的框架，该框架基于微调前后安全关键神经元的相似性差异，恢复了安全关键神经元的安全性。我们提出了一种基于微调前后安全关键神经元的相似性差异来恢复LLMS安全性的免训练框架。我们框架的核心是首先从最初对齐的模型构建安全参考模型，以放大神经元中与安全相关的特征。然后，我们利用这个参考模型来识别安全关键神经元，我们将其准备为补丁。最后，我们通过移植这些准备好的补丁，选择性地只恢复那些表现出显著相似性差异的神经元，从而最大限度地改变微调的模型。广泛的实验表明，在跨多个下游任务的微调模型中，显著增强了安全性，同时极大地保持了任务级的准确性。我们的发现表明，一些安全关键神经元的区域在微调后显示出明显的差异，这种差异可以通过从参考模型移植神经元来有效纠正，而不需要额外的训练。代码将位于\url{https://github.com/xinykou/NLSR}



## **11. Adversarial Attacks on Large Language Models in Medicine**

医学中对大型语言模型的对抗攻击 cs.AI

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2406.12259v3) [paper-pdf](http://arxiv.org/pdf/2406.12259v3)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.

摘要: 将大型语言模型(LLM)集成到医疗保健应用程序中，在医疗诊断、治疗建议和患者护理方面提供了有希望的进步。然而，LLMS对对抗性攻击的敏感性构成了一个重大威胁，可能会在微妙的医疗环境中导致有害后果。本研究调查了LLMS在三个医疗任务中对两种类型的对抗性攻击的脆弱性。利用真实世界的患者数据，我们证明了开源和专有LLM都容易受到跨多个任务的操纵。这项研究进一步表明，特定领域的任务在模型微调中需要比一般领域任务更多的对抗性数据才能有效地执行攻击，特别是对于能力更强的模型。我们发现，虽然整合对抗性数据并不会显著降低医学基准上的整体模型性能，但它确实会导致微调模型权重的显著变化，这表明了一条检测和对抗模型攻击的潜在路径。这项研究强调了迫切需要强有力的安全措施和开发防御机制来保护医疗应用中的低成本管理，以确保其在医疗保健环境中的安全和有效部署。



## **12. When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations**

当后门说话时：通过模型生成的解释了解LLM后门攻击 cs.CR

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2411.12701v2) [paper-pdf](http://arxiv.org/pdf/2411.12701v2)

**Authors**: Huaizhi Ge, Yiming Li, Qifan Wang, Yongfeng Zhang, Ruixiang Tang

**Abstract**: Large Language Models (LLMs) are known to be vulnerable to backdoor attacks, where triggers embedded in poisoned samples can maliciously alter LLMs' behaviors. In this paper, we move beyond attacking LLMs and instead examine backdoor attacks through the novel lens of natural language explanations. Specifically, we leverage LLMs' generative capabilities to produce human-readable explanations for their decisions, enabling direct comparisons between explanations for clean and poisoned samples. Our results show that backdoored models produce coherent explanations for clean inputs but diverse and logically flawed explanations for poisoned data, a pattern consistent across classification and generation tasks for different backdoor attacks. Further analysis reveals key insights into the explanation generation process. At the token level, explanation tokens associated with poisoned samples only appear in the final few transformer layers. At the sentence level, attention dynamics indicate that poisoned inputs shift attention away from the original input context during explanation generation. These findings enhance our understanding of backdoor mechanisms in LLMs and present a promising framework for detecting vulnerabilities through explainability.

摘要: 众所周知，大型语言模型(LLM)容易受到后门攻击，在后门攻击中，嵌入在有毒样本中的触发器可以恶意改变LLM的行为。在这篇文章中，我们超越了攻击LLM，而是通过自然语言解释的新视角来研究后门攻击。具体地说，我们利用LLMS的生成能力来为他们的决定生成人类可读的解释，从而能够在干净和有毒样本的解释之间进行直接比较。我们的结果表明，回溯模型为干净的输入提供了连贯的解释，但对有毒数据提供了多样化和逻辑上有缺陷的解释，对于不同的后门攻击，这种模式在分类和生成任务中是一致的。进一步的分析揭示了对解释生成过程的关键见解。在令牌级别，与中毒样本相关的解释令牌只出现在最后几个变压器层中。在句子层面，注意动力学表明，有毒输入在解释生成过程中将注意力从原始输入上下文转移开。这些发现加深了我们对LLMS中后门机制的理解，并为通过可解释性检测漏洞提供了一个很有前途的框架。



## **13. Stepwise Reasoning Error Disruption Attack of LLMs**

LLM的逐步推理错误中断攻击 cs.AI

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11934v1) [paper-pdf](http://arxiv.org/pdf/2412.11934v1)

**Authors**: Jingyu Peng, Maolin Wang, Xiangyu Zhao, Kai Zhang, Wanyu Wang, Pengyue Jia, Qidong Liu, Ruocheng Guo, Qi Liu

**Abstract**: Large language models (LLMs) have made remarkable strides in complex reasoning tasks, but their safety and robustness in reasoning processes remain underexplored. Existing attacks on LLM reasoning are constrained by specific settings or lack of imperceptibility, limiting their feasibility and generalizability. To address these challenges, we propose the Stepwise rEasoning Error Disruption (SEED) attack, which subtly injects errors into prior reasoning steps to mislead the model into producing incorrect subsequent reasoning and final answers. Unlike previous methods, SEED is compatible with zero-shot and few-shot settings, maintains the natural reasoning flow, and ensures covert execution without modifying the instruction. Extensive experiments on four datasets across four different models demonstrate SEED's effectiveness, revealing the vulnerabilities of LLMs to disruptions in reasoning processes. These findings underscore the need for greater attention to the robustness of LLM reasoning to ensure safety in practical applications.

摘要: 大语言模型在复杂的推理任务中取得了显著的进展，但其在推理过程中的安全性和稳健性仍未得到充分的研究。现有的对LLM推理的攻击受到特定环境或缺乏不可见性的限制，限制了它们的可行性和普适性。为了应对这些挑战，我们提出了逐步推理错误中断(SEED)攻击，它巧妙地在先前的推理步骤中注入错误，以误导模型产生不正确的后续推理和最终答案。与以往的方法不同，SEED兼容零射和少射设置，保持了自然的推理流程，在不修改指令的情况下确保了隐蔽的执行。在四个不同模型的四个数据集上的广泛实验证明了SEED的有效性，揭示了LLMS在推理过程中对中断的脆弱性。这些发现强调了需要更多地关注LLM推理的健壮性，以确保在实际应用中的安全性。



## **14. PBI-Attack: Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for Toxicity Maximization**

PBI攻击：优先引导双峰交互黑匣子越狱攻击，以实现毒性最大化 cs.CR

Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for  Toxicity Maximization

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.05892v2) [paper-pdf](http://arxiv.org/pdf/2412.05892v2)

**Authors**: Ruoxi Cheng, Yizhong Ding, Shuirong Cao, Ranjie Duan, Xiaoshuang Jia, Shaowei Yuan, Zhiqiang Wang, Xiaojun Jia

**Abstract**: Understanding the vulnerabilities of Large Vision Language Models (LVLMs) to jailbreak attacks is essential for their responsible real-world deployment. Most previous work requires access to model gradients, or is based on human knowledge (prompt engineering) to complete jailbreak, and they hardly consider the interaction of images and text, resulting in inability to jailbreak in black box scenarios or poor performance. To overcome these limitations, we propose a Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for toxicity maximization, referred to as PBI-Attack. Our method begins by extracting malicious features from a harmful corpus using an alternative LVLM and embedding these features into a benign image as prior information. Subsequently, we enhance these features through bidirectional cross-modal interaction optimization, which iteratively optimizes the bimodal perturbations in an alternating manner through greedy search, aiming to maximize the toxicity of the generated response. The toxicity level is quantified using a well-trained evaluation model. Experiments demonstrate that PBI-Attack outperforms previous state-of-the-art jailbreak methods, achieving an average attack success rate of 92.5% across three open-source LVLMs and around 67.3% on three closed-source LVLMs. Disclaimer: This paper contains potentially disturbing and offensive content.

摘要: 了解大型视觉语言模型(LVLM)对越狱攻击的脆弱性对于它们在现实世界中负责任的部署至关重要。以前的工作大多需要获取模型梯度，或者基于人类知识(提示工程)来完成越狱，并且很少考虑图像和文本的交互，导致在黑匣子场景下无法越狱或性能不佳。为了克服这些局限性，我们提出了一种先验引导的双模交互黑盒越狱攻击，称为PBI攻击。我们的方法首先使用替代的LVLM从有害语料库中提取恶意特征，并将这些特征作为先验信息嵌入到良性图像中。随后，我们通过双向跨模式交互优化来增强这些特征，该优化通过贪婪搜索以交替的方式迭代优化双峰扰动，以最大化所生成响应的毒性。使用训练有素的评估模型来量化毒性水平。实验表明，PBI-Attack的性能优于以往最先进的越狱方法，在三个开源LVLM上的平均攻击成功率为92.5%，在三个闭源LVLM上的平均攻击成功率约为67.3%。免责声明：本文包含可能令人不安和冒犯性的内容。



## **15. Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks**

克服一切困难：克服多语言嵌入倒置攻击中的类型学、脚本和语言混乱 cs.CL

11 pages, 4 figures, 7 tables

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2408.11749v2) [paper-pdf](http://arxiv.org/pdf/2408.11749v2)

**Authors**: Yiyi Chen, Russa Biswas, Heather Lent, Johannes Bjerva

**Abstract**: Large Language Models (LLMs) are susceptible to malicious influence by cyber attackers through intrusions such as adversarial, backdoor, and embedding inversion attacks. In response, the burgeoning field of LLM Security aims to study and defend against such threats. Thus far, the majority of works in this area have focused on monolingual English models, however, emerging research suggests that multilingual LLMs may be more vulnerable to various attacks than their monolingual counterparts. While previous work has investigated embedding inversion over a small subset of European languages, it is challenging to extrapolate these findings to languages from different linguistic families and with differing scripts. To this end, we explore the security of multilingual LLMs in the context of embedding inversion attacks and investigate cross-lingual and cross-script inversion across 20 languages, spanning over 8 language families and 12 scripts. Our findings indicate that languages written in Arabic script and Cyrillic script are particularly vulnerable to embedding inversion, as are languages within the Indo-Aryan language family. We further observe that inversion models tend to suffer from language confusion, sometimes greatly reducing the efficacy of an attack. Accordingly, we systematically explore this bottleneck for inversion models, uncovering predictable patterns which could be leveraged by attackers. Ultimately, this study aims to further the field's understanding of the outstanding security vulnerabilities facing multilingual LLMs and raise awareness for the languages most at risk of negative impact from these attacks.

摘要: 大型语言模型(LLM)容易受到网络攻击者通过对抗性、后门和嵌入反转攻击等入侵的恶意影响。作为回应，LLM Security这个新兴领域的目标是研究和防御此类威胁。到目前为止，这一领域的研究大多集中在单语英语模型上，然而，新的研究表明，多语种的LLM可能比单语的LLM更容易受到各种攻击。虽然以前的工作已经研究了在一小部分欧洲语言上嵌入倒置，但将这些发现外推到来自不同语系和不同脚本的语言是具有挑战性的。为此，我们在嵌入倒置攻击的情况下探索了多语言LLMS的安全性，并研究了跨语言和跨脚本的跨语言和跨脚本倒置，涉及8个语系和12个脚本。我们的发现表明，用阿拉伯文字和西里尔文字书写的语言特别容易嵌入倒置，印度-雅利安语系的语言也是如此。我们进一步观察到，倒置模型往往受到语言混乱的影响，有时会极大地降低攻击的有效性。因此，我们系统地探索了倒置模型的这一瓶颈，揭示了可被攻击者利用的可预测模式。最终，这项研究旨在加深外地对多语种土地管理系统面临的突出安全漏洞的了解，并提高对最有可能受到这些攻击的负面影响的语言的认识。



## **16. Intention Analysis Makes LLMs A Good Jailbreak Defender**

意图分析使LLC成为出色的越狱捍卫者 cs.CL

COLING 2025

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2401.06561v4) [paper-pdf](http://arxiv.org/pdf/2401.06561v4)

**Authors**: Yuqi Zhang, Liang Ding, Lefei Zhang, Dacheng Tao

**Abstract**: Aligning large language models (LLMs) with human values, particularly when facing complex and stealthy jailbreak attacks, presents a formidable challenge. Unfortunately, existing methods often overlook this intrinsic nature of jailbreaks, which limits their effectiveness in such complex scenarios. In this study, we present a simple yet highly effective defense strategy, i.e., Intention Analysis ($\mathbb{IA}$). $\mathbb{IA}$ works by triggering LLMs' inherent self-correct and improve ability through a two-stage process: 1) analyzing the essential intention of the user input, and 2) providing final policy-aligned responses based on the first round conversation. Notably, $\mathbb{IA}$ is an inference-only method, thus could enhance LLM safety without compromising their helpfulness. Extensive experiments on varying jailbreak benchmarks across a wide range of LLMs show that $\mathbb{IA}$ could consistently and significantly reduce the harmfulness in responses (averagely -48.2% attack success rate). Encouragingly, with our $\mathbb{IA}$, Vicuna-7B even outperforms GPT-3.5 regarding attack success rate. We empirically demonstrate that, to some extent, $\mathbb{IA}$ is robust to errors in generated intentions. Further analyses reveal the underlying principle of $\mathbb{IA}$: suppressing LLM's tendency to follow jailbreak prompts, thereby enhancing safety.

摘要: 要使大型语言模型(LLM)与人类价值观保持一致，尤其是在面临复杂而隐蔽的越狱攻击时，这是一个艰巨的挑战。不幸的是，现有的方法往往忽略了越狱的这一内在本质，这限制了它们在如此复杂的情况下的有效性。在本研究中，我们提出了一种简单而高效的防御策略，即意图分析($\mathbb{IA}$)。$\mathbb{IA}$的工作原理是通过两个阶段触发LLM固有的自我纠正和改进能力：1)分析用户输入的基本意图，2)基于第一轮对话提供最终的策略一致的响应。值得注意的是，$\mathbb{IA}$是一种仅限推理的方法，因此可以在不影响其有用性的情况下增强LLM的安全性。对不同越狱基准的广泛实验表明，$\mathbb{IA}$可以持续且显著地降低响应中的危害性(平均攻击成功率为-48.2%)。令人鼓舞的是，有了我们的$\mathbb{IA}$，维库纳-7B在攻击成功率方面甚至超过了GPT-3.5。我们的经验证明，在某种程度上，$\mathbb{IA}$对于生成意图中的错误是健壮的。进一步的分析揭示了$\mathbb{IA}$的基本原理：抑制LLM遵循越狱提示的倾向，从而增强安全性。



## **17. Revisiting Backdoor Attacks against Large Vision-Language Models from Domain Shift**

重新审视领域转移对大型视觉语言模型的后门攻击 cs.CV

11 pages, 9 figures

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2406.18844v4) [paper-pdf](http://arxiv.org/pdf/2406.18844v4)

**Authors**: Siyuan Liang, Jiawei Liang, Tianyu Pang, Chao Du, Aishan Liu, Mingli Zhu, Xiaochun Cao, Dacheng Tao

**Abstract**: Instruction tuning enhances large vision-language models (LVLMs) but increases their vulnerability to backdoor attacks due to their open design. Unlike prior studies in static settings, this paper explores backdoor attacks in LVLM instruction tuning across mismatched training and testing domains. We introduce a new evaluation dimension, backdoor domain generalization, to assess attack robustness under visual and text domain shifts. Our findings reveal two insights: (1) backdoor generalizability improves when distinctive trigger patterns are independent of specific data domains or model architectures, and (2) the competitive interaction between trigger patterns and clean semantic regions, where guiding the model to predict triggers enhances attack generalizability. Based on these insights, we propose a multimodal attribution backdoor attack (MABA) that injects domain-agnostic triggers into critical areas using attributional interpretation. Experiments with OpenFlamingo, Blip-2, and Otter show that MABA significantly boosts the attack success rate of generalization by 36.4%, achieving a 97% success rate at a 0.2% poisoning rate. This study reveals limitations in current evaluations and highlights how enhanced backdoor generalizability poses a security threat to LVLMs, even without test data access.

摘要: 指令调优增强了大型视觉语言模型(LVLM)，但由于其开放式设计，增加了它们对后门攻击的脆弱性。与以往在静态设置下的研究不同，本文探讨了在不匹配的训练域和测试域中的LVLM指令调优中的后门攻击。我们引入了一个新的评估维度--后门领域泛化，来评估视觉和文本领域变化下的攻击健壮性。对OpenFlamingo、BLIP-2和Otter的实验表明，Maba显著提高了泛化攻击成功率36.4%，在0.2%的投毒率下获得了97%的成功率。



## **18. Exploiting the Index Gradients for Optimization-Based Jailbreaking on Large Language Models**

利用索引要素对大型语言模型进行基于优化的越狱 cs.CL

13 pages,2 figures, accepted by COLING 2025

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.08615v2) [paper-pdf](http://arxiv.org/pdf/2412.08615v2)

**Authors**: Jiahui Li, Yongchang Hao, Haoyu Xu, Xing Wang, Yu Hong

**Abstract**: Despite the advancements in training Large Language Models (LLMs) with alignment techniques to enhance the safety of generated content, these models remain susceptible to jailbreak, an adversarial attack method that exposes security vulnerabilities in LLMs. Notably, the Greedy Coordinate Gradient (GCG) method has demonstrated the ability to automatically generate adversarial suffixes that jailbreak state-of-the-art LLMs. However, the optimization process involved in GCG is highly time-consuming, rendering the jailbreaking pipeline inefficient. In this paper, we investigate the process of GCG and identify an issue of Indirect Effect, the key bottleneck of the GCG optimization. To this end, we propose the Model Attack Gradient Index GCG (MAGIC), that addresses the Indirect Effect by exploiting the gradient information of the suffix tokens, thereby accelerating the procedure by having less computation and fewer iterations. Our experiments on AdvBench show that MAGIC achieves up to a 1.5x speedup, while maintaining Attack Success Rates (ASR) on par or even higher than other baselines. Our MAGIC achieved an ASR of 74% on the Llama-2 and an ASR of 54% when conducting transfer attacks on GPT-3.5. Code is available at https://github.com/jiah-li/magic.

摘要: 尽管在使用对齐技术训练大型语言模型(LLM)以增强生成内容的安全性方面取得了进展，但这些模型仍然容易受到越狱的影响，这是一种暴露LLM安全漏洞的对抗性攻击方法。值得注意的是，贪婪坐标梯度(GCG)方法已经展示了自动生成敌意后缀的能力，这些后缀是越狱最先进的LLM。然而，GCG涉及的优化过程非常耗时，使得越狱管道效率低下。在本文中，我们研究了GCG的过程，并找出了间接影响的问题，这是GCG优化的关键瓶颈。为此，我们提出了模型攻击梯度索引GCG(MAGIC)，它通过利用后缀标记的梯度信息来解决间接影响，从而以更少的计算量和更少的迭代来加速过程。我们在AdvBtch上的实验表明，Magic在保持攻击成功率(ASR)与其他基线相当甚至更高的情况下，实现了高达1.5倍的加速。我们的魔法在骆驼-2上达到了74%的ASR，当对GPT-3.5进行传输攻击时ASR达到54%。代码可在https://github.com/jiah-li/magic.上找到



## **19. Failures to Find Transferable Image Jailbreaks Between Vision-Language Models**

未能在视觉语言模型之间找到可传输的图像越狱 cs.CL

NeurIPS 2024 Workshops: RBFM (Best Paper), Frontiers in AdvML (Oral),  Red Teaming GenAI (Oral), SoLaR (Spotlight), SATA

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2407.15211v2) [paper-pdf](http://arxiv.org/pdf/2407.15211v2)

**Authors**: Rylan Schaeffer, Dan Valentine, Luke Bailey, James Chua, Cristóbal Eyzaguirre, Zane Durante, Joe Benton, Brando Miranda, Henry Sleight, John Hughes, Rajashree Agrawal, Mrinank Sharma, Scott Emmons, Sanmi Koyejo, Ethan Perez

**Abstract**: The integration of new modalities into frontier AI systems offers exciting capabilities, but also increases the possibility such systems can be adversarially manipulated in undesirable ways. In this work, we focus on a popular class of vision-language models (VLMs) that generate text outputs conditioned on visual and textual inputs. We conducted a large-scale empirical study to assess the transferability of gradient-based universal image ``jailbreaks" using a diverse set of over 40 open-parameter VLMs, including 18 new VLMs that we publicly release. Overall, we find that transferable gradient-based image jailbreaks are extremely difficult to obtain. When an image jailbreak is optimized against a single VLM or against an ensemble of VLMs, the jailbreak successfully jailbreaks the attacked VLM(s), but exhibits little-to-no transfer to any other VLMs; transfer is not affected by whether the attacked and target VLMs possess matching vision backbones or language models, whether the language model underwent instruction-following and/or safety-alignment training, or many other factors. Only two settings display partially successful transfer: between identically-pretrained and identically-initialized VLMs with slightly different VLM training data, and between different training checkpoints of a single VLM. Leveraging these results, we then demonstrate that transfer can be significantly improved against a specific target VLM by attacking larger ensembles of ``highly-similar" VLMs. These results stand in stark contrast to existing evidence of universal and transferable text jailbreaks against language models and transferable adversarial attacks against image classifiers, suggesting that VLMs may be more robust to gradient-based transfer attacks.

摘要: 将新的模式集成到前沿人工智能系统中提供了令人兴奋的能力，但也增加了此类系统被以不受欢迎的方式进行相反操作的可能性。在这项工作中，我们专注于一类流行的视觉语言模型(VLM)，它们生成以视觉和文本输入为条件的文本输出。我们进行了一项大规模的实证研究，以评估基于梯度的通用图像“越狱”的可转移性，使用了40多个开放参数VLM的不同集合，其中包括我们公开发布的18个新的VLM。总体而言，我们发现基于梯度的可转移越狱图像非常难以获得。只有两个设置显示部分成功的传输：在具有略微不同的VLM训练数据的相同预训练和相同初始化的VLM之间，以及在单个VLM的不同训练检查点之间。利用这些结果，我们随后证明了针对特定目标VLm的转移可以通过攻击更大的“高度相似的”VLM集合来显著改善。这些结果与针对语言模型的普遍和可传输的文本越狱以及针对图像分类器的可传输的对抗性攻击的现有证据形成了鲜明对比，这表明VLM可能对基于梯度的传输攻击更健壮。



## **20. Finding a Wolf in Sheep's Clothing: Combating Adversarial Text-To-Image Prompts with Text Summarization**

披着羊皮找狼：用文本摘要对抗对抗文本到图像的对抗性冲突 cs.CR

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.12212v1) [paper-pdf](http://arxiv.org/pdf/2412.12212v1)

**Authors**: Portia Cooper, Harshita Narnoli, Mihai Surdeanu

**Abstract**: Text-to-image models are vulnerable to the stepwise "Divide-and-Conquer Attack" (DACA) that utilize a large language model to obfuscate inappropriate content in prompts by wrapping sensitive text in a benign narrative. To mitigate stepwise DACA attacks, we propose a two-layer method involving text summarization followed by binary classification. We assembled the Adversarial Text-to-Image Prompt (ATTIP) dataset ($N=940$), which contained DACA-obfuscated and non-obfuscated prompts. From the ATTIP dataset, we created two summarized versions: one generated by a small encoder model and the other by a large language model. Then, we used an encoder classifier and a GPT-4o classifier to perform content moderation on the summarized and unsummarized prompts. When compared with a classifier that operated over the unsummarized data, our method improved F1 score performance by 31%. Further, the highest recorded F1 score achieved (98%) was produced by the encoder classifier on a summarized ATTIP variant. This study indicates that pre-classification text summarization can inoculate content detection models against stepwise DACA obfuscations.

摘要: 为了缓解分步式DACA攻击，我们提出了一种文本摘要和二进制分类相结合的两层方法。从ATTIP数据集，我们创建了两个汇总版本：一个由小型编码器模型生成，另一个由大型语言模型生成。然后，我们使用编码器分类器和GPT-40分类器对摘要和未摘要的提示进行内容审核。与处理未汇总数据的分类器相比，我们的方法将F1得分性能提高了31%。此外，获得的最高记录F1分数(98%)是由编码员在总结的ATTIP变体上产生的。这项研究表明，预分类文本摘要可以为内容检测模型接种针对逐步DACA混淆的疫苗。



## **21. Red Teaming GPT-4V: Are GPT-4V Safe Against Uni/Multi-Modal Jailbreak Attacks?**

Red Teaming GPT-4V：GPT-4V可以安全对抗Uni/Multi-Modal越狱攻击吗？ cs.LG

technical report; update code repo link

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2404.03411v2) [paper-pdf](http://arxiv.org/pdf/2404.03411v2)

**Authors**: Shuo Chen, Zhen Han, Bailan He, Zifeng Ding, Wenqian Yu, Philip Torr, Volker Tresp, Jindong Gu

**Abstract**: Various jailbreak attacks have been proposed to red-team Large Language Models (LLMs) and revealed the vulnerable safeguards of LLMs. Besides, some methods are not limited to the textual modality and extend the jailbreak attack to Multimodal Large Language Models (MLLMs) by perturbing the visual input. However, the absence of a universal evaluation benchmark complicates the performance reproduction and fair comparison. Besides, there is a lack of comprehensive evaluation of closed-source state-of-the-art (SOTA) models, especially MLLMs, such as GPT-4V. To address these issues, this work first builds a comprehensive jailbreak evaluation dataset with 1445 harmful questions covering 11 different safety policies. Based on this dataset, extensive red-teaming experiments are conducted on 11 different LLMs and MLLMs, including both SOTA proprietary models and open-source models. We then conduct a deep analysis of the evaluated results and find that (1) GPT4 and GPT-4V demonstrate better robustness against jailbreak attacks compared to open-source LLMs and MLLMs. (2) Llama2 and Qwen-VL-Chat are more robust compared to other open-source models. (3) The transferability of visual jailbreak methods is relatively limited compared to textual jailbreak methods. The dataset and code can be found https://github.com/chenxshuo/RedTeamingGPT4V

摘要: 此外，一些方法并不局限于文本情态，通过扰动视觉输入将越狱攻击扩展到多模式大语言模型(MLLMS)。然而，由于没有通用的评价基准，业绩复制和公平比较变得更加复杂。此外，对闭源最先进(SOTA)模型，特别是MLLMS，如GPT-4V，缺乏全面的评估。为了解决这些问题，这项工作首先建立了一个全面的越狱评估数据集，其中包含1445个有害问题，涵盖11种不同的安全政策。基于这个数据集，在11个不同的LLM和MLLM上进行了广泛的红团队实验，包括Sota专有模型和开源模型。然后我们对评估结果进行了深入的分析，发现(1)GPT4和GPT-4V与开源的LLMS和MLLMS相比，对越狱攻击表现出更好的健壮性。(2)与其他开源模型相比，Llama2和Qwen-VL-Chat的健壮性更强。数据集和代码可在https://github.com/chenxshuo/RedTeamingGPT4V中找到



## **22. The Superalignment of Superhuman Intelligence with Large Language Models**

超人智能与大型语言模型的超级对齐 cs.CL

Under review of Science China

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.11145v1) [paper-pdf](http://arxiv.org/pdf/2412.11145v1)

**Authors**: Minlie Huang, Yingkang Wang, Shiyao Cui, Pei Ke, Jie Tang

**Abstract**: We have witnessed superhuman intelligence thanks to the fast development of large language models and multimodal language models. As the application of such superhuman models becomes more and more common, a critical question rises here: how can we ensure superhuman models are still safe, reliable and aligned well to human values? In this position paper, we discuss the concept of superalignment from the learning perspective to answer this question by outlining the learning paradigm shift from large-scale pretraining, supervised fine-tuning, to alignment training. We define superalignment as designing effective and efficient alignment algorithms to learn from noisy-labeled data (point-wise samples or pair-wise preference data) in a scalable way when the task becomes very complex for human experts to annotate and the model is stronger than human experts. We highlight some key research problems in superalignment, namely, weak-to-strong generalization, scalable oversight, and evaluation. We then present a conceptual framework for superalignment, which consists of three modules: an attacker which generates adversary queries trying to expose the weaknesses of a learner model; a learner which will refine itself by learning from scalable feedbacks generated by a critic model along with minimal human experts; and a critic which generates critics or explanations for a given query-response pair, with a target of improving the learner by criticizing. We discuss some important research problems in each component of this framework and highlight some interesting research ideas that are closely related to our proposed framework, for instance, self-alignment, self-play, self-refinement, and more. Last, we highlight some future research directions for superalignment, including identification of new emergent risks and multi-dimensional alignment.

摘要: [TencentCloudSDKException] code:InternalError message:Translation failed requestId:589eac85-6485-43a6-9e38-9d339df7a8b3



## **23. Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models**

通过扩散模型高效生成视觉语言模型的有针对性且可转移的对抗示例 cs.CV

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2404.10335v4) [paper-pdf](http://arxiv.org/pdf/2404.10335v4)

**Authors**: Qi Guo, Shanmin Pang, Xiaojun Jia, Yang Liu, Qing Guo

**Abstract**: Adversarial attacks, particularly \textbf{targeted} transfer-based attacks, can be used to assess the adversarial robustness of large visual-language models (VLMs), allowing for a more thorough examination of potential security flaws before deployment. However, previous transfer-based adversarial attacks incur high costs due to high iteration counts and complex method structure. Furthermore, due to the unnaturalness of adversarial semantics, the generated adversarial examples have low transferability. These issues limit the utility of existing methods for assessing robustness. To address these issues, we propose AdvDiffVLM, which uses diffusion models to generate natural, unrestricted and targeted adversarial examples via score matching. Specifically, AdvDiffVLM uses Adaptive Ensemble Gradient Estimation to modify the score during the diffusion model's reverse generation process, ensuring that the produced adversarial examples have natural adversarial targeted semantics, which improves their transferability. Simultaneously, to improve the quality of adversarial examples, we use the GradCAM-guided Mask method to disperse adversarial semantics throughout the image rather than concentrating them in a single area. Finally, AdvDiffVLM embeds more target semantics into adversarial examples after multiple iterations. Experimental results show that our method generates adversarial examples 5x to 10x faster than state-of-the-art transfer-based adversarial attacks while maintaining higher quality adversarial examples. Furthermore, compared to previous transfer-based adversarial attacks, the adversarial examples generated by our method have better transferability. Notably, AdvDiffVLM can successfully attack a variety of commercial VLMs in a black-box environment, including GPT-4V.

摘要: [TencentCloudSDKException] code:InternalError message:Translation failed requestId:a57b1c90-32d9-42bf-84f3-df7b03885351



## **24. Do Chase Your Tail! Missing Key Aspects Augmentation in Textual Vulnerability Descriptions of Long-tail Software through Feature Inference**

一定要追你的尾巴！通过特征推理增强长尾软件文本漏洞描述中缺失的关键方面 cs.SE

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2405.07430v2) [paper-pdf](http://arxiv.org/pdf/2405.07430v2)

**Authors**: Linyi Han, Shidong Pan, Zhenchang Xing, Jiamou Sun, Sofonias Yitagesu, Xiaowang Zhang, Zhiyong Feng

**Abstract**: Augmenting missing key aspects in Textual Vulnerability Descriptions (TVDs) is crucial for effective vulnerability analysis. For instance, in TVDs, key aspects include Attack Vector, Vulnerability Type, among others. These key aspects help security engineers understand and address the vulnerability in a timely manner. For software with a large user base (non-long-tail software), augmenting these missing key aspects has significantly advanced vulnerability analysis and software security research. However, software instances with a limited user base (long-tail software) often get overlooked due to inconsistency software names, TVD limited avaliability, and domain-specific jargon, which complicates vulnerability analysis and software repairs. In this paper, we introduce a novel software feature inference framework designed to augment the missing key aspects of TVDs for long-tail software. Firstly, we tackle the issue of non-standard software names found in community-maintained vulnerability databases by cross-referencing government databases with Common Vulnerabilities and Exposures (CVEs). Next, we employ Large Language Models (LLMs) to generate the missing key aspects. However, the limited availability of historical TVDs restricts the variety of examples. To overcome this limitation, we utilize the Common Weakness Enumeration (CWE) to classify all TVDs and select cluster centers as representative examples. To ensure accuracy, we present Natural Language Inference (NLI) models specifically designed for long-tail software. These models identify and eliminate incorrect responses. Additionally, we use a wiki repository to provide explanations for proprietary terms.

摘要: 例如，在TVDS中，关键方面包括攻击矢量、漏洞类型等。这些关键方面帮助安全工程师及时了解和解决漏洞。对于拥有大量用户基础的软件(非长尾软件)，补充这些缺失的关键方面将极大地促进漏洞分析和软件安全研究。然而，由于软件名称不一致、TVD可用性有限以及特定于域的术语，用户基础有限的软件实例(长尾软件)经常被忽略，这会使漏洞分析和软件修复复杂化。在本文中，我们介绍了一种新的软件特征推理框架，旨在为长尾软件补充TVDS中缺失的关键方面。首先，我们通过交叉引用具有公共漏洞和暴露(CVE)的政府数据库来解决社区维护的漏洞数据库中发现的非标准软件名称的问题。接下来，我们使用大型语言模型(LLM)来生成缺少的关键方面。为了克服这一局限性，我们利用共同弱点枚举(CWE)对所有的TV进行分类，并选择集群中心作为代表性的例子。为了确保准确性，我们提出了专门为长尾软件设计的自然语言推理(NLI)模型。这些模型可以识别并消除不正确的回答。此外，我们使用维基存储库为专有术语提供解释。



## **25. Simulate and Eliminate: Revoke Backdoors for Generative Large Language Models**

模拟和消除：撤销生成性大型语言模型的后门 cs.CR

To appear at AAAI 2025

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2405.07667v2) [paper-pdf](http://arxiv.org/pdf/2405.07667v2)

**Authors**: Haoran Li, Yulin Chen, Zihao Zheng, Qi Hu, Chunkit Chan, Heshan Liu, Yangqiu Song

**Abstract**: With rapid advances, generative large language models (LLMs) dominate various Natural Language Processing (NLP) tasks from understanding to reasoning. Yet, language models' inherent vulnerabilities may be exacerbated due to increased accessibility and unrestricted model training on massive data. A malicious adversary may publish poisoned data online and conduct backdoor attacks on the victim LLMs pre-trained on the poisoned data. Backdoored LLMs behave innocuously for normal queries and generate harmful responses when the backdoor trigger is activated. Despite significant efforts paid to LLMs' safety issues, LLMs are still struggling against backdoor attacks. As Anthropic recently revealed, existing safety training strategies, including supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), fail to revoke the backdoors once the LLM is backdoored during the pre-training stage. In this paper, we present Simulate and Eliminate (SANDE) to erase the undesired backdoored mappings for generative LLMs. We initially propose Overwrite Supervised Fine-tuning (OSFT) for effective backdoor removal when the trigger is known. Then, to handle scenarios where trigger patterns are unknown, we integrate OSFT into our two-stage framework, SANDE. Unlike other works that assume access to cleanly trained models, our safety-enhanced LLMs are able to revoke backdoors without any reference. Consequently, our safety-enhanced LLMs no longer produce targeted responses when the backdoor triggers are activated. We conduct comprehensive experiments to show that our proposed SANDE is effective against backdoor attacks while bringing minimal harm to LLMs' powerful capability.

摘要: 随着研究的深入，从理解到推理的各种自然语言处理任务都被生成性大语言模型(LLMS)所支配。然而，语言模型的固有脆弱性可能会由于海量数据的可获得性增加和不受限制的模型训练而加剧。恶意对手可能会在网上发布有毒数据，并对受害者LLM进行后门攻击，这些LLM预先训练了有毒数据。后门LLM在正常查询中的行为是无害的，并在激活后门触发器时生成有害的响应。尽管在LLMS的安全问题上付出了巨大的努力，但LLMS仍在努力应对后门攻击。正如人类最近揭示的那样，现有的安全培训策略，包括监督微调(SFT)和从人类反馈的强化学习(RLHF)，一旦LLM在培训前阶段后退，就无法取消后门。在这篇文章中，我们提出了模拟和消除(SANDE)来消除生成式LLMS中不需要的回溯映射。我们最初提出了覆盖监督精调(OSFT)，用于在已知触发器的情况下有效地删除后门。然后，为了处理触发模式未知的场景，我们将OSFT集成到我们的两阶段框架Sande中。与其他假定可以访问训练有素的模型的作品不同，我们的安全增强型LLM能够在没有任何参考的情况下撤销后门。因此，当后门触发器被激活时，我们的安全增强型LLMS不再产生目标响应。我们进行了全面的实验，以表明我们提出的SANDE能够有效地抵抗后门攻击，同时对LLMS的强大能力造成的损害最小。



## **26. Separate the Wheat from the Chaff: A Post-Hoc Approach to Safety Re-Alignment for Fine-Tuned Language Models**

将小麦与谷壳分开：精调语言模型的安全重新对齐的事后方法 cs.CL

14 pages, 12 figures,

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.11041v1) [paper-pdf](http://arxiv.org/pdf/2412.11041v1)

**Authors**: Di Wu, Xin Lu, Yanyan Zhao, Bing Qin

**Abstract**: Although large language models (LLMs) achieve effective safety alignment at the time of release, they still face various safety challenges. A key issue is that fine-tuning often compromises the safety alignment of LLMs. To address this issue, we propose a method named \textbf{IRR} (\textbf{I}dentify, \textbf{R}emove, and \textbf{R}ecalibrate for Safety Realignment) that performs safety realignment for LLMs. The core of IRR is to identify and remove unsafe delta parameters from the fine-tuned models, while recalibrating the retained ones. We evaluate the effectiveness of IRR across various datasets, including both full fine-tuning and LoRA methods. Our results demonstrate that IRR significantly enhances the safety performance of fine-tuned models on safety benchmarks, such as harmful queries and jailbreak attacks, while maintaining their performance on downstream tasks. The source code is available at: \url{https://anonymous.4open.science/r/IRR-BD4F}.

摘要: 尽管大型语言模型（LLM）在发布时实现了有效的安全一致，但它们仍然面临各种安全挑战。一个关键问题是微调通常会损害LLM的安全对齐。为了解决这个问题，我们提出了一种名为\textBF{IRR}（\textBF{I}识别、\textBF{R}emove和\textBF{R}重新校准）的方法，该方法为LLM执行安全调整。IRR的核心是从微调模型中识别并删除不安全的Delta参数，同时重新校准保留的参数。我们评估IRR在各种数据集的有效性，包括完全微调和LoRA方法。我们的结果表明，IRR显着增强了经过微调的模型在安全基准（例如有害查询和越狱攻击）上的安全性能，同时保持了其在下游任务上的性能。源代码可访问：\url{https：//anonymous.4open.science/r/IRR-BD 4F}。



## **27. Labeling NIDS Rules with MITRE ATT&CK Techniques: Machine Learning vs. Large Language Models**

使用MITRE ATT & CK技术标记NIDS规则：机器学习与大型语言模型 cs.CR

**SubmitDate**: 2024-12-14    [abs](http://arxiv.org/abs/2412.10978v1) [paper-pdf](http://arxiv.org/pdf/2412.10978v1)

**Authors**: Nir Daniel, Florian Klaus Kaiser, Shay Giladi, Sapir Sharabi, Raz Moyal, Shalev Shpolyansky, Andres Murillo, Aviad Elyashar, Rami Puzis

**Abstract**: Analysts in Security Operations Centers (SOCs) are often occupied with time-consuming investigations of alerts from Network Intrusion Detection Systems (NIDS). Many NIDS rules lack clear explanations and associations with attack techniques, complicating the alert triage and the generation of attack hypotheses. Large Language Models (LLMs) may be a promising technology to reduce the alert explainability gap by associating rules with attack techniques. In this paper, we investigate the ability of three prominent LLMs (ChatGPT, Claude, and Gemini) to reason about NIDS rules while labeling them with MITRE ATT&CK tactics and techniques. We discuss prompt design and present experiments performed with 973 Snort rules. Our results indicate that while LLMs provide explainable, scalable, and efficient initial mappings, traditional Machine Learning (ML) models consistently outperform them in accuracy, achieving higher precision, recall, and F1-scores. These results highlight the potential for hybrid LLM-ML approaches to enhance SOC operations and better address the evolving threat landscape.

摘要: 许多NIDS规则缺乏明确的解释和与攻击技术的关联，使警报分类和攻击假设的生成复杂化。通过将规则与攻击技术相关联，大型语言模型(LLM)可能是一种很有前途的技术，可以缩小警报可解释性差距。在这篇文章中，我们调查了三个重要的LLM(ChatGPT，Claude和Gemini)在用MITRE ATT&CK策略和技术来标记它们时，对NIDS规则进行推理的能力。我们讨论了即时设计，并给出了使用973 Snort规则执行的实验。我们的结果表明，虽然LLMS提供了可解释的、可扩展的和有效的初始映射，但传统的机器学习(ML)模型在准确率上始终优于它们，获得了更高的准确率、召回率和F1分数。这些结果突出了混合LLM-ML方法在增强SOC操作和更好地应对不断变化的威胁环境方面的潜力。



## **28. CEKER: A Generalizable LLM Framework for Literature Analysis with a Case Study in Unikernel Security**

CEKER：一个可推广的LLM文献分析框架，并附有Unikernell Security案例研究 cs.CR

7 pages, 2 figures

**SubmitDate**: 2024-12-14    [abs](http://arxiv.org/abs/2412.10904v1) [paper-pdf](http://arxiv.org/pdf/2412.10904v1)

**Authors**: Alex Wollman, John Hastings

**Abstract**: Literature reviews are a critical component of formulating and justifying new research, but are a manual and often time-consuming process. This research introduces a novel, generalizable approach to literature analysis called CEKER which uses a three-step process to streamline the collection of literature, the extraction of key insights, and the summarized analysis of key trends and gaps. Leveraging Large Language Models (LLMs), this methodology represents a significant shift from traditional manual literature reviews, offering a scalable, flexible, and repeatable approach that can be applied across diverse research domains.   A case study on unikernel security illustrates CEKER's ability to generate novel insights validated against previous manual methods. CEKER's analysis highlighted reduced attack surface as the most prominent theme. Key security gaps included the absence of Address Space Layout Randomization, missing debugging tools, and limited entropy generation, all of which represent important challenges to unikernel security. The study also revealed a reliance on hypervisors as a potential attack vector and emphasized the need for dynamic security adjustments to address real-time threats.

摘要: 文献综述是形成和证明新研究的关键组成部分，但它是一个手动的、往往很耗时的过程。这项研究介绍了一种新的、可概括的文学分析方法，称为CEKER，它使用三个步骤来简化文献收集、关键见解的提取以及关键趋势和差距的汇总分析。利用大型语言模型(LLM)，该方法代表了与传统手动文献审查的重大转变，提供了一种可扩展、灵活和可重复的方法，可应用于不同的研究领域。一个关于单内核安全的案例研究展示了Ceker生成与以前的手动方法相比较的新见解的能力。切克的分析强调，减少攻击面是最突出的主题。关键的安全漏洞包括缺乏地址空间布局随机化、缺少调试工具以及有限的熵生成，所有这些都是对单内核安全的重要挑战。该研究还揭示了对管理程序作为潜在攻击载体的依赖，并强调了动态安全调整的必要性，以应对实时威胁。



## **29. Towards Action Hijacking of Large Language Model-based Agent**

基于大型语言模型的Agent的动作劫持 cs.CR

**SubmitDate**: 2024-12-14    [abs](http://arxiv.org/abs/2412.10807v1) [paper-pdf](http://arxiv.org/pdf/2412.10807v1)

**Authors**: Yuyang Zhang, Kangjie Chen, Xudong Jiang, Yuxiang Sun, Run Wang, Lina Wang

**Abstract**: In the past few years, intelligent agents powered by large language models (LLMs) have achieved remarkable progress in performing complex tasks. These LLM-based agents receive queries as tasks and decompose them into various subtasks via the equipped LLMs to guide the action of external entities (\eg{}, tools, AI-agents) to answer the questions from users. Empowered by their exceptional capabilities of understanding and problem-solving, they are widely adopted in labor-intensive sectors including healthcare, finance, code completion, \etc{} At the same time, there are also concerns about the potential misuse of these agents, prompting the built-in safety guards from service providers. To circumvent the built-in guidelines, the prior studies proposed a multitude of attacks including memory poisoning, jailbreak, and prompt injection. These studies often fail to maintain effectiveness across safety filters employed by agents due to the restricted privileges and the harmful semantics in queries. In this paper, we introduce \Name, a novel hijacking attack to manipulate the action plans of black-box agent system. \Name first collects the action-aware memory through prompt theft from long-term memory. It then leverages the internal memory retrieval mechanism of the agent to provide an erroneous context. The huge gap between the latent spaces of the retriever and safety filters allows our method to bypass the detection easily. Extensive experimental results demonstrate the effectiveness of our apporach (\eg{}, 99.67\% ASR). Besides, our approach achieved an average bypass rate of 92.7\% for safety filters.

摘要: 这些基于LLM的代理将查询作为任务接收，并通过配备的LLMS将其分解为各个子任务，以指导外部实体(如{}、工具、AI-Agents)回答用户的问题。为了绕过内置的指导方针，先前的研究提出了一系列攻击，包括记忆中毒、越狱和快速注射。由于受限的特权和查询中的有害语义，这些研究经常无法保持对代理所使用的安全过滤器的有效性。在本文中，我们介绍了一种新的劫持攻击--NAME，它可以操纵黑盒代理系统的行动计划。\NAME首先通过从长期记忆中即时窃取来收集动作感知记忆。然后，它利用代理的内部存储器检索机制来提供错误的上下文。检索器的潜在空间和安全过滤器之间的巨大差距使得我们的方法可以很容易地绕过检测。大量的实验结果证明了该算法的有效性(例如，99.67ASR)。此外，该方法对安全过滤器的平均旁通率为92.7%。



## **30. On Effects of Steering Latent Representation for Large Language Model Unlearning**

论引导潜在表示对大型语言模型取消学习的影响 cs.CL

Accepted at AAAI-25 Main Technical Track

**SubmitDate**: 2024-12-14    [abs](http://arxiv.org/abs/2408.06223v2) [paper-pdf](http://arxiv.org/pdf/2408.06223v2)

**Authors**: Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue

**Abstract**: Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. We investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. We show that RMU unlearned models are robust against adversarial jailbreak attacks. Furthermore, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU -- a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.

摘要: 遗忘表征误导(RMU)是一种有效的大语言模型遗忘方法，它将中间层的模型表征引导到目标随机表征。尽管其表现良好，但其根本原因和解释仍未得到充分研究。在本文中，我们从理论上证明了中间层中的转向遗忘表征降低了令牌置信度，从而导致LLM产生错误或无意义的响应。我们研究了系数如何影响遗忘样本表示与随机方向的对齐，并提示了跨不同网络层有效遗忘的最优系数值。我们证明了RMU未学习模型对敌意越狱攻击是健壮的。此外，我们的实证分析表明，当RMU应用于LLMS的中后期时，其有效性较差。为了解决这一缺陷，我们提出了自适应RMU--一种简单但有效的替代方法，使遗忘在大多数层都有效。大量实验表明，与现有技术相比，自适应RMU在不增加额外计算代价的情况下，显著改善了遗忘性能。



## **31. No Free Lunch for Defending Against Prefilling Attack by In-Context Learning**

通过上下文学习防御预填充攻击没有免费午餐 cs.CR

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2412.12192v1) [paper-pdf](http://arxiv.org/pdf/2412.12192v1)

**Authors**: Zhiyu Xue, Guangliang Liu, Bocheng Chen, Kristen Marie Johnson, Ramtin Pedarsani

**Abstract**: The security of Large Language Models (LLMs) has become an important research topic since the emergence of ChatGPT. Though there have been various effective methods to defend against jailbreak attacks, prefilling attacks remain an unsolved and popular threat against open-sourced LLMs. In-Context Learning (ICL) offers a computationally efficient defense against various jailbreak attacks, yet no effective ICL methods have been developed to counter prefilling attacks. In this paper, we: (1) show that ICL can effectively defend against prefilling jailbreak attacks by employing adversative sentence structures within demonstrations; (2) characterize the effectiveness of this defense through the lens of model size, number of demonstrations, over-defense, integration with other jailbreak attacks, and the presence of safety alignment. Given the experimental results and our analysis, we conclude that there is no free lunch for defending against prefilling jailbreak attacks with ICL. On the one hand, current safety alignment methods fail to mitigate prefilling jailbreak attacks, but adversative structures within ICL demonstrations provide robust defense across various model sizes and complex jailbreak attacks. On the other hand, LLMs exhibit similar over-defensiveness when utilizing ICL demonstrations with adversative structures, and this behavior appears to be independent of model size.

摘要: 自ChatGPT出现以来，大语言模型的安全性就成为一个重要的研究课题。虽然已经有各种有效的方法来防御越狱攻击，但预填充攻击仍然是对开源LLM的一个尚未解决的普遍威胁。上下文学习(ICL)提供了一种针对各种越狱攻击的计算高效的防御方法，但还没有开发出有效的ICL方法来对抗预填充攻击。在本文中，我们：(1)证明了ICL能够有效地防御预先填充越狱攻击；(2)从模型大小、演示次数、过度防御、与其他越狱攻击的集成以及安全对齐的存在等方面表征了这种防御的有效性。根据实验结果和我们的分析，我们得出结论：用ICL防御预填充式越狱攻击不存在免费午餐。一方面，当前的安全对齐方法无法缓解预填充越狱攻击，但ICL演示中的对抗性结构在各种模型大小和复杂的越狱攻击中提供了强大的防御。另一方面，当使用具有转折结构的ICL演示时，LLM表现出类似的过度防御，并且这种行为似乎与模型大小无关。



## **32. AdvPrefix: An Objective for Nuanced LLM Jailbreaks**

AdvPreFix：细致入微的LLM越狱目标 cs.LG

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2412.10321v1) [paper-pdf](http://arxiv.org/pdf/2412.10321v1)

**Authors**: Sicheng Zhu, Brandon Amos, Yuandong Tian, Chuan Guo, Ivan Evtimov

**Abstract**: Many jailbreak attacks on large language models (LLMs) rely on a common objective: making the model respond with the prefix "Sure, here is (harmful request)". While straightforward, this objective has two limitations: limited control over model behaviors, often resulting in incomplete or unrealistic responses, and a rigid format that hinders optimization. To address these limitations, we introduce AdvPrefix, a new prefix-forcing objective that enables more nuanced control over model behavior while being easy to optimize. Our objective leverages model-dependent prefixes, automatically selected based on two criteria: high prefilling attack success rates and low negative log-likelihood. It can further simplify optimization by using multiple prefixes for a single user request. AdvPrefix can integrate seamlessly into existing jailbreak attacks to improve their performance for free. For example, simply replacing GCG attack's target prefixes with ours on Llama-3 improves nuanced attack success rates from 14% to 80%, suggesting that current alignment struggles to generalize to unseen prefixes. Our work demonstrates the importance of jailbreak objectives in achieving nuanced jailbreaks.

摘要: 许多针对大型语言模型(LLM)的越狱攻击都依赖于一个共同的目标：让该模型以“当然，这是(有害的请求)”作为响应前缀。虽然很简单，但这个目标有两个限制：对模型行为的有限控制，通常会导致不完整或不现实的响应，以及阻碍优化的僵化格式。为了解决这些限制，我们引入了AdvPrefix，这是一种新的前缀强制目标，可以对模型行为进行更细微的控制，同时易于优化。我们的目标是利用依赖于模型的前缀，基于两个标准自动选择：高预填充攻击成功率和低负对数似然。它可以通过对单个用户请求使用多个前缀来进一步简化优化。AdvPrefix可以无缝集成到现有的越狱攻击中，以免费提高它们的性能。例如，简单地用我们在Llama-3上的目标前缀替换GCG攻击的目标前缀，可以将细微差别的攻击成功率从14%提高到80%，这表明当前的排列难以推广到看不见的前缀。我们的工作证明了越狱目标在实现细微差别越狱方面的重要性。



## **33. RTL-Breaker: Assessing the Security of LLMs against Backdoor Attacks on HDL Code Generation**

RTL-Breaker：评估LLM的安全性，以应对对HDL代码生成的后门攻击 cs.CR

Accepted at 2025 Design, Automation & Test in Europe (DATE)  Conference

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2411.17569v2) [paper-pdf](http://arxiv.org/pdf/2411.17569v2)

**Authors**: Lakshmi Likhitha Mankali, Jitendra Bhandari, Manaar Alam, Ramesh Karri, Michail Maniatakos, Ozgur Sinanoglu, Johann Knechtel

**Abstract**: Large language models (LLMs) have demonstrated remarkable potential with code generation/completion tasks for hardware design. In fact, LLM-based hardware description language (HDL) code generation has enabled the industry to realize complex designs more quickly, reducing the time and effort required in the development cycle. However, the increased reliance on such automation introduces critical security risks. Notably, given that LLMs have to be trained on vast datasets of codes that are typically sourced from publicly available repositories (often without thorough validation), LLMs are susceptible to so-called data poisoning or backdoor attacks. Here, attackers inject malicious code for the training data, which can be carried over into the HDL code generated by LLMs. This threat vector can compromise the security and integrity of entire hardware systems. In this work, we propose RTL-Breaker, a novel backdoor attack framework on LLM-based HDL code generation. RTL-Breaker provides an in-depth analysis for essential aspects of this novel problem: 1) various trigger mechanisms versus their effectiveness for inserting malicious modifications, and 2) side-effects by backdoor attacks on code generation in general, i.e., impact on code quality. RTL-Breaker emphasizes the urgent need for more robust measures to safeguard against such attacks. Toward that end, we open-source our framework and all data.

摘要: 大型语言模型(LLM)在硬件设计的代码生成/完成任务方面表现出了巨大的潜力。事实上，基于LLM的硬件描述语言(HDL)代码生成使业界能够更快地实现复杂的设计，减少了开发周期所需的时间和精力。然而，对这种自动化的日益依赖带来了严重的安全风险。值得注意的是，鉴于LLM必须接受大量代码数据集的培训，这些代码通常来自公开可用的存储库(通常没有进行彻底验证)，因此LLM很容易受到所谓的数据中毒或后门攻击。在这里，攻击者为训练数据注入恶意代码，这些代码可以被带到LLMS生成的HDL码中。这种威胁媒介可能会危及整个硬件系统的安全性和完整性。在这项工作中，我们提出了一种新的后门攻击框架RTL-Breaker，用于基于LLM的硬件描述语言代码生成。RTL-Breaker提供了对这个新问题的基本方面的深入分析：1)各种触发机制与其插入恶意修改的有效性的对比，以及2)后门攻击对代码生成的一般副作用，即对代码质量的影响。RTL-Breaker强调迫切需要采取更强有力的措施来防范此类攻击。为此，我们将我们的框架和所有数据开源。



## **34. From Allies to Adversaries: Manipulating LLM Tool-Calling through Adversarial Injection**

从盟友到对手：通过对抗注入操纵LLM工具调用 cs.CR

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2412.10198v1) [paper-pdf](http://arxiv.org/pdf/2412.10198v1)

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang

**Abstract**: Tool-calling has changed Large Language Model (LLM) applications by integrating external tools, significantly enhancing their functionality across diverse tasks. However, this integration also introduces new security vulnerabilities, particularly in the tool scheduling mechanisms of LLM, which have not been extensively studied. To fill this gap, we present ToolCommander, a novel framework designed to exploit vulnerabilities in LLM tool-calling systems through adversarial tool injection. Our framework employs a well-designed two-stage attack strategy. Firstly, it injects malicious tools to collect user queries, then dynamically updates the injected tools based on the stolen information to enhance subsequent attacks. These stages enable ToolCommander to execute privacy theft, launch denial-of-service attacks, and even manipulate business competition by triggering unscheduled tool-calling. Notably, the ASR reaches 91.67% for privacy theft and hits 100% for denial-of-service and unscheduled tool calling in certain cases. Our work demonstrates that these vulnerabilities can lead to severe consequences beyond simple misuse of tool-calling systems, underscoring the urgent need for robust defensive strategies to secure LLM Tool-calling systems.

摘要: 工具调用通过集成外部工具改变了大型语言模型(LLM)应用程序，显著增强了它们在不同任务中的功能。然而，这种集成也引入了新的安全漏洞，特别是在LLM的工具调度机制中，这些漏洞还没有得到广泛的研究。为了填补这一空白，我们提出了一种新的框架，它旨在通过敌意工具注入来利用LLM工具调用系统中的漏洞。我们的框架采用了精心设计的两阶段攻击策略。它首先注入恶意工具来收集用户查询，然后根据窃取的信息动态更新注入的工具，以加强后续攻击。这些阶段使工具指挥官能够执行隐私窃取、发起拒绝服务攻击，甚至通过触发计划外的工具调用来操纵业务竞争。值得注意的是，在某些情况下，隐私窃取的ASR达到91.67%，拒绝服务和非计划工具调用的ASR达到100%。我们的工作表明，这些漏洞可能导致严重后果，而不仅仅是简单地滥用工具调用系统，这突显了迫切需要强大的防御战略来保护LLM工具调用系统。



## **35. Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders**

值得信赖的LLM：使用知识库和双解码器定制和基础文本生成 cs.CL

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2411.07870v5) [paper-pdf](http://arxiv.org/pdf/2411.07870v5)

**Authors**: Xiaofeng Zhu, Jaya Krishna Mandivarapu

**Abstract**: Although people are impressed by the content generation skills of large language models, the use of LLMs, such as ChatGPT, is limited by the domain grounding of the content. The correctness and groundedness of the generated content need to be based on a verified context, such as results from Retrieval-Augmented Generation (RAG). One important issue when adapting LLMs to a customized domain is that the generated responses are often incomplete, or the additions are not verified and may even be hallucinated. Prior studies on hallucination detection have focused on evaluation metrics, which are not easily adaptable to dynamic domains and can be vulnerable to attacks like jail-breaking. In this work, we propose 1) a post-processing algorithm that leverages knowledge triplets in RAG context to correct hallucinations and 2) a dual-decoder model that fuses RAG context to guide the generation process.

摘要: 尽管人们对大型语言模型的内容生成技能印象深刻，但ChatGPT等LLM的使用受到内容领域基础的限制。生成内容的正确性和可信度需要基于经过验证的上下文，例如检索增强生成（RAG）的结果。将LLM调整到定制域时的一个重要问题是，生成的响应通常不完整，或者添加未经验证，甚至可能出现幻觉。之前关于幻觉检测的研究集中在评估指标上，这些指标不容易适应动态领域，并且容易受到越狱等攻击。在这项工作中，我们提出了1）一种利用RAG上下文中的知识三重组来纠正幻觉的后处理算法，以及2）一种融合RAG上下文来指导生成过程的双解码器模型。



## **36. AdvWave: Stealthy Adversarial Jailbreak Attack against Large Audio-Language Models**

AdvWave：针对大型音频语言模型的隐形对抗越狱攻击 cs.SD

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08608v1) [paper-pdf](http://arxiv.org/pdf/2412.08608v1)

**Authors**: Mintong Kang, Chejian Xu, Bo Li

**Abstract**: Recent advancements in large audio-language models (LALMs) have enabled speech-based user interactions, significantly enhancing user experience and accelerating the deployment of LALMs in real-world applications. However, ensuring the safety of LALMs is crucial to prevent risky outputs that may raise societal concerns or violate AI regulations. Despite the importance of this issue, research on jailbreaking LALMs remains limited due to their recent emergence and the additional technical challenges they present compared to attacks on DNN-based audio models. Specifically, the audio encoders in LALMs, which involve discretization operations, often lead to gradient shattering, hindering the effectiveness of attacks relying on gradient-based optimizations. The behavioral variability of LALMs further complicates the identification of effective (adversarial) optimization targets. Moreover, enforcing stealthiness constraints on adversarial audio waveforms introduces a reduced, non-convex feasible solution space, further intensifying the challenges of the optimization process. To overcome these challenges, we develop AdvWave, the first jailbreak framework against LALMs. We propose a dual-phase optimization method that addresses gradient shattering, enabling effective end-to-end gradient-based optimization. Additionally, we develop an adaptive adversarial target search algorithm that dynamically adjusts the adversarial optimization target based on the response patterns of LALMs for specific queries. To ensure that adversarial audio remains perceptually natural to human listeners, we design a classifier-guided optimization approach that generates adversarial noise resembling common urban sounds. Extensive evaluations on multiple advanced LALMs demonstrate that AdvWave outperforms baseline methods, achieving a 40% higher average jailbreak attack success rate.

摘要: 大型音频语言模型(LALM)的最新进展实现了基于语音的用户交互，显著增强了用户体验，并加快了LALM在现实世界应用中的部署。然而，确保LALM的安全对于防止可能引发社会担忧或违反人工智能法规的高风险输出至关重要。尽管这一问题很重要，但由于最近出现的LALM以及与攻击基于DNN的音频模型相比带来的额外技术挑战，对越狱LALM的研究仍然有限。具体地说，LALMS中的音频编码器涉及离散化操作，经常会导致梯度破碎，阻碍了依赖于基于梯度优化的攻击的有效性。LALMS的行为变异性使有效(对抗性)优化目标的识别变得更加复杂。此外，对敌方音频波形实施隐蔽性约束会引入一个缩减的非凸可行解空间，从而进一步加剧了优化过程的挑战。为了克服这些挑战，我们开发了第一个针对LALMS的越狱框架AdvWave。我们提出了一种双阶段优化方法，解决了梯度破碎问题，实现了有效的端到端基于梯度的优化。此外，我们还开发了一种自适应对抗性目标搜索算法，该算法根据LALMS对特定查询的响应模式动态调整对抗性优化目标。为了确保对抗性音频对人类听众来说保持感知上的自然，我们设计了一种分类器引导的优化方法，该方法产生类似于常见城市声音的对抗性噪声。对多个高级LALM的广泛评估表明，AdvWave的性能优于基准方法，实现了40%的平均越狱攻击成功率。



## **37. Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts**

彩虹团队：开放式一代的多元化对抗预言 cs.CL

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2402.16822v3) [paper-pdf](http://arxiv.org/pdf/2402.16822v3)

**Authors**: Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rocktäschel, Roberta Raileanu

**Abstract**: As large language models (LLMs) become increasingly prevalent across many real-world applications, understanding and enhancing their robustness to adversarial attacks is of paramount importance. Existing methods for identifying adversarial prompts tend to focus on specific domains, lack diversity, or require extensive human annotations. To address these limitations, we present Rainbow Teaming, a novel black-box approach for producing a diverse collection of adversarial prompts. Rainbow Teaming casts adversarial prompt generation as a quality-diversity problem and uses open-ended search to generate prompts that are both effective and diverse. Focusing on the safety domain, we use Rainbow Teaming to target various state-of-the-art LLMs, including the Llama 2 and Llama 3 models. Our approach reveals hundreds of effective adversarial prompts, with an attack success rate exceeding 90% across all tested models. Furthermore, we demonstrate that prompts generated by Rainbow Teaming are highly transferable and that fine-tuning models with synthetic data generated by our method significantly enhances their safety without sacrificing general performance or helpfulness. We additionally explore the versatility of Rainbow Teaming by applying it to question answering and cybersecurity, showcasing its potential to drive robust open-ended self-improvement in a wide range of applications.

摘要: 随着大型语言模型(LLM)在许多真实世界的应用中变得越来越普遍，理解和增强它们对对手攻击的健壮性是至关重要的。现有的识别对抗性提示的方法往往集中在特定的领域，缺乏多样性，或者需要大量的人工注释。为了解决这些局限性，我们提出了彩虹分组，这是一种新的黑盒方法，用于产生多样化的对抗性提示集合。彩虹团队将敌意提示生成视为质量多样性问题，并使用开放式搜索来生成既有效又多样化的提示。专注于安全领域，我们使用彩虹团队瞄准各种最先进的LLM，包括Llama 2和Llama 3型号。我们的方法揭示了数百个有效的对抗性提示，在所有测试模型上的攻击成功率超过90%。此外，我们证明了彩虹组合生成的提示具有很高的可转移性，并且使用我们的方法生成的合成数据对模型进行微调显著地增强了它们的安全性，而不会牺牲一般性能或帮助。我们还探讨了彩虹团队的多功能性，将其应用于问题回答和网络安全，展示了其在广泛应用中推动强大的开放式自我改进的潜力。



## **38. Underestimated Privacy Risks for Minority Populations in Large Language Model Unlearning**

大型语言模型放弃学习中少数族裔的隐私风险被低估 cs.LG

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08559v1) [paper-pdf](http://arxiv.org/pdf/2412.08559v1)

**Authors**: Rongzhe Wei, Mufei Li, Mohsen Ghassemi, Eleonora Kreačić, Yifan Li, Xiang Yue, Bo Li, Vamsi K. Potluru, Pan Li, Eli Chien

**Abstract**: Large Language Models are trained on extensive datasets that often contain sensitive, human-generated information, raising significant concerns about privacy breaches. While certified unlearning approaches offer strong privacy guarantees, they rely on restrictive model assumptions that are not applicable to LLMs. As a result, various unlearning heuristics have been proposed, with the associated privacy risks assessed only empirically. The standard evaluation pipelines typically randomly select data for removal from the training set, apply unlearning techniques, and use membership inference attacks to compare the unlearned models against models retrained without the to-be-unlearned data. However, since every data point is subject to the right to be forgotten, unlearning should be considered in the worst-case scenario from the privacy perspective. Prior work shows that data outliers may exhibit higher memorization effects. Intuitively, they are harder to be unlearn and thus the privacy risk of unlearning them is underestimated in the current evaluation. In this paper, we leverage minority data to identify such a critical flaw in previously widely adopted evaluations. We substantiate this claim through carefully designed experiments, including unlearning canaries related to minority groups, inspired by privacy auditing literature. Using personally identifiable information as a representative minority identifier, we demonstrate that minority groups experience at least 20% more privacy leakage in most cases across six unlearning approaches, three MIAs, three benchmark datasets, and two LLMs of different scales. Given that the right to be forgotten should be upheld for every individual, we advocate for a more rigorous evaluation of LLM unlearning methods. Our minority-aware evaluation framework represents an initial step toward ensuring more equitable assessments of LLM unlearning efficacy.

摘要: 大型语言模型在大量数据集上进行训练，这些数据集通常包含敏感的人类生成的信息，这引发了人们对侵犯隐私的严重担忧。虽然经过认证的遗忘方法提供了强大的隐私保证，但它们依赖于不适用于LLM的限制性模型假设。因此，各种遗忘启发式方法被提出，相关的隐私风险仅通过经验进行评估。标准评估流水线通常随机选择要从训练集中移除的数据，应用遗忘技术，并使用成员关系推理攻击来将未学习的模型与没有待学习数据的重新训练的模型进行比较。然而，由于每个数据点都有被遗忘的权利，所以从隐私的角度来看，遗忘应该在最坏的情况下考虑。前人的工作表明，数据离群点可能会表现出更高的记忆效果。直觉上，它们更难被忘记，因此在当前的评估中，忘记它们的隐私风险被低估了。在这篇文章中，我们利用少数群体数据来识别以前被广泛采用的评估中的这样一个关键缺陷。我们通过精心设计的实验来证实这一说法，包括在隐私审计文献的启发下，忘记与少数群体有关的金丝雀。使用个人可识别信息作为代表性的少数群体识别符，我们证明在六种遗忘方法、三个MIA、三个基准数据集和两个不同规模的LLMS中，少数群体在大多数情况下经历的隐私泄露至少多20%。鉴于每个人都应该维护被遗忘的权利，我们主张对LLM遗忘方法进行更严格的评估。我们的少数群体意识评估框架是朝着确保对LLM遗忘效能进行更公平的评估迈出的第一步。



## **39. Model-Editing-Based Jailbreak against Safety-aligned Large Language Models**

针对安全性一致的大型语言模型的基于模型编辑的越狱 cs.CR

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08201v1) [paper-pdf](http://arxiv.org/pdf/2412.08201v1)

**Authors**: Yuxi Li, Zhibo Zhang, Kailong Wang, Ling Shi, Haoyu Wang

**Abstract**: Large Language Models (LLMs) have transformed numerous fields by enabling advanced natural language interactions but remain susceptible to critical vulnerabilities, particularly jailbreak attacks. Current jailbreak techniques, while effective, often depend on input modifications, making them detectable and limiting their stealth and scalability. This paper presents Targeted Model Editing (TME), a novel white-box approach that bypasses safety filters by minimally altering internal model structures while preserving the model's intended functionalities. TME identifies and removes safety-critical transformations (SCTs) embedded in model matrices, enabling malicious queries to bypass restrictions without input modifications. By analyzing distinct activation patterns between safe and unsafe queries, TME isolates and approximates SCTs through an optimization process. Implemented in the D-LLM framework, our method achieves an average Attack Success Rate (ASR) of 84.86% on four mainstream open-source LLMs, maintaining high performance. Unlike existing methods, D-LLM eliminates the need for specific triggers or harmful response collections, offering a stealthier and more effective jailbreak strategy. This work reveals a covert and robust threat vector in LLM security and emphasizes the need for stronger safeguards in model safety alignment.

摘要: 大型语言模型(LLM)通过实现高级自然语言交互改变了许多领域，但仍然容易受到关键漏洞的影响，特别是越狱攻击。目前的越狱技术虽然有效，但往往依赖于输入修改，使其可被检测，并限制了其隐蔽性和可扩展性。本文提出了目标模型编辑(TME)，这是一种新的白盒方法，它通过最小限度地改变内部模型结构来绕过安全过滤器，同时保持模型的预期功能。TME可识别并删除模型矩阵中嵌入的安全关键转换(SCT)，从而使恶意查询无需修改输入即可绕过限制。通过分析安全和不安全查询之间的不同激活模式，TME通过优化过程分离和近似SCT。该方法在D-LLM框架下实现，在四个主流开源LLM上的平均攻击成功率(ASR)达到84.86%，保持了高性能。与现有方法不同，D-LLM消除了对特定触发器或有害响应收集的需要，提供了更隐蔽和更有效的越狱策略。这项工作揭示了LLM安全中的隐蔽和强大的威胁向量，并强调了在模型安全调整中需要更强大的保障措施。



## **40. Doubly-Universal Adversarial Perturbations: Deceiving Vision-Language Models Across Both Images and Text with a Single Perturbation**

双重普遍对抗性扰动：通过单一扰动欺骗图像和文本的视觉语言模型 cs.CV

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08108v1) [paper-pdf](http://arxiv.org/pdf/2412.08108v1)

**Authors**: Hee-Seon Kim, Minbeom Kim, Changick Kim

**Abstract**: Large Vision-Language Models (VLMs) have demonstrated remarkable performance across multimodal tasks by integrating vision encoders with large language models (LLMs). However, these models remain vulnerable to adversarial attacks. Among such attacks, Universal Adversarial Perturbations (UAPs) are especially powerful, as a single optimized perturbation can mislead the model across various input images. In this work, we introduce a novel UAP specifically designed for VLMs: the Doubly-Universal Adversarial Perturbation (Doubly-UAP), capable of universally deceiving VLMs across both image and text inputs. To successfully disrupt the vision encoder's fundamental process, we analyze the core components of the attention mechanism. After identifying value vectors in the middle-to-late layers as the most vulnerable, we optimize Doubly-UAP in a label-free manner with a frozen model. Despite being developed as a black-box to the LLM, Doubly-UAP achieves high attack success rates on VLMs, consistently outperforming baseline methods across vision-language tasks. Extensive ablation studies and analyses further demonstrate the robustness of Doubly-UAP and provide insights into how it influences internal attention mechanisms.

摘要: 大视觉语言模型(VLM)通过将视觉编码器与大语言模型(LLM)相结合，在多通道任务中表现出了显著的性能。然而，这些模型仍然容易受到对手的攻击。在这些攻击中，通用对抗性扰动(UAP)尤其强大，因为单个优化的扰动可以在不同的输入图像上误导模型。在这项工作中，我们介绍了一种新的专门针对VLMS设计的UAP：双重通用对抗性摄动(Double-Universal Aversarial微扰，Double-UAP)，能够在图像和文本输入之间普遍欺骗VLMS。为了成功地扰乱视觉编码器的基本过程，我们分析了注意机制的核心组件。在确定中后期价值向量最易受攻击后，我们使用冻结模型以无标签的方式对Double-UAP进行优化。尽管被开发为LLM的黑匣子，Double-UAP在VLM上实现了高攻击成功率，在视觉语言任务中始终优于基线方法。广泛的消融研究和分析进一步证明了Double-UAP的健壮性，并提供了对其如何影响内部注意机制的见解。



## **41. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

时间序列预测大型语言模型中的对抗漏洞 cs.LG

11 pages, 5 figures

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08099v1) [paper-pdf](http://arxiv.org/pdf/2412.08099v1)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in the field of time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like TimeGPT and LLM-Time with GPT-3.5, GPT-4, LLaMa, and Mistral, show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications.

摘要: 大型语言模型最近在时间序列预测领域显示出巨大的潜力，在处理复杂的时间数据方面提供了令人印象深刻的能力。然而，它们在实际应用中的健壮性和可靠性仍然没有得到充分的研究，特别是关于它们对对手攻击的敏感性。本文提出了一种基于LLM的时间序列预测的对抗性攻击框架。通过使用无梯度和黑盒优化方法，我们产生了最小但高效的扰动，这些扰动显著降低了跨多个数据集和LLM体系结构的预测精度。我们的实验，包括使用GPT-3.5、GPT-4、LLAMA和Mistral的TimeGPT和LLM-Time模型，表明对抗性攻击导致的性能降级比随机噪声严重得多，并证明了我们的攻击在不同LLM上的广泛有效性。这些结果强调了低层管理在时间序列预测中的关键弱点，强调了需要强大的防御机制来确保其在实际应用中的可靠部署。



## **42. What You See Is Not Always What You Get: An Empirical Study of Code Comprehension by Large Language Models**

你所看到的并不总是你所得到的：大型语言模型对代码理解的实证研究 cs.SE

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08098v1) [paper-pdf](http://arxiv.org/pdf/2412.08098v1)

**Authors**: Bangshuo Zhu, Jiawen Wen, Huaming Chen

**Abstract**: Recent studies have demonstrated outstanding capabilities of large language models (LLMs) in software engineering domain, covering numerous tasks such as code generation and comprehension. While the benefit of LLMs for coding task is well noted, it is perceived that LLMs are vulnerable to adversarial attacks. In this paper, we study the specific LLM vulnerability to imperceptible character attacks, a type of prompt-injection attack that uses special characters to befuddle an LLM whilst keeping the attack hidden to human eyes. We devise four categories of attacks and investigate their effects on the performance outcomes of tasks relating to code analysis and code comprehension. Two generations of ChatGPT are included to evaluate the impact of advancements made to contemporary models. Our experimental design consisted of comparing perturbed and unperturbed code snippets and evaluating two performance outcomes, which are model confidence using log probabilities of response, and correctness of response. We conclude that earlier version of ChatGPT exhibits a strong negative linear correlation between the amount of perturbation and the performance outcomes, while the recent ChatGPT presents a strong negative correlation between the presence of perturbation and performance outcomes, but no valid correlational relationship between perturbation budget and performance outcomes. We anticipate this work contributes to an in-depth understanding of leveraging LLMs for coding tasks. It is suggested future research should delve into how to create LLMs that can return a correct response even if the prompt exhibits perturbations.

摘要: 最近的研究表明，大型语言模型(LLM)在软件工程领域具有卓越的能力，涵盖了代码生成和理解等众多任务。虽然LLMS对于编码任务的好处是众所周知的，但人们认为LLMS容易受到对手的攻击。在本文中，我们研究了特定的LLM对隐蔽字符攻击的脆弱性，这是一种使用特殊字符来迷惑LLM的即时注入攻击，同时将攻击隐藏在人眼之外。我们设计了四类攻击，并调查了它们对与代码分析和代码理解相关的任务的性能结果的影响。两代ChatGPT被包括在内，以评估当代模型所取得的进步的影响。我们的实验设计包括比较扰动和未扰动的代码片段，并评估两个性能结果，即使用响应的对数概率的模型置信度和响应的正确性。我们的结论是，早期版本的ChatGPT表现出扰动量与绩效结果之间的强负相关，而最近版本的ChatGPT呈现出扰动的存在与绩效结果之间的强负相关，但扰动预算与绩效结果之间没有有效的相关关系。我们预计这项工作有助于深入理解利用LLM进行编码任务。有人建议，未来的研究应该深入研究如何创建即使在提示显示扰动的情况下也能返回正确响应的LLM。



## **43. Plentiful Jailbreaks with String Compositions**

弦乐作品丰富越狱 cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2411.01084v3) [paper-pdf](http://arxiv.org/pdf/2411.01084v3)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or red-teamers, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary string compositions, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.

摘要: 大型语言模型（LLM）仍然容易受到一系列对抗攻击和越狱方法的影响。白帽攻击者或红团队使用的一种常见方法是使用字符串级混淆处理模型输入和输出，其中可以包括leetspeak、旋转密码、Base 64、ASC等。我们的工作通过将这些基于编码的攻击统一到可逆字符串转换的框架中来扩展它们。通过可逆性，我们可以设计任意的字符串组合，定义为转换序列，我们可以通过编程方式进行端到端编码和解码。我们设计了一种自动化的n中最佳攻击，该攻击从组合上大量的字符串组合中进行采样。在HarmBench上进行评估时，我们的越狱在几个领先的前沿模型上获得了有竞争力的攻击成功率，这凸显了即使在高级LLM中，基于编码的攻击仍然是一个持久的漏洞。



## **44. Summon a Demon and Bind it: A Grounded Theory of LLM Red Teaming**

召唤恶魔并绑定它：法学硕士红色团队的接地理论 cs.CL

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2311.06237v3) [paper-pdf](http://arxiv.org/pdf/2311.06237v3)

**Authors**: Nanna Inie, Jonathan Stray, Leon Derczynski

**Abstract**: Engaging in the deliberate generation of abnormal outputs from Large Language Models (LLMs) by attacking them is a novel human activity. This paper presents a thorough exposition of how and why people perform such attacks, defining LLM red-teaming based on extensive and diverse evidence. Using a formal qualitative methodology, we interviewed dozens of practitioners from a broad range of backgrounds, all contributors to this novel work of attempting to cause LLMs to fail. We focused on the research questions of defining LLM red teaming, uncovering the motivations and goals for performing the activity, and characterizing the strategies people use when attacking LLMs. Based on the data, LLM red teaming is defined as a limit-seeking, non-malicious, manual activity, which depends highly on a team-effort and an alchemist mindset. It is highly intrinsically motivated by curiosity, fun, and to some degrees by concerns for various harms of deploying LLMs. We identify a taxonomy of 12 strategies and 35 different techniques of attacking LLMs. These findings are presented as a comprehensive grounded theory of how and why people attack large language models: LLM red teaming.

摘要: 通过攻击大语言模型来刻意生成异常输出是一种新的人类活动。这篇文章详细阐述了人们如何以及为什么进行这样的攻击，并基于广泛和多样化的证据定义了LLM红队。使用正式的定性方法，我们采访了来自广泛背景的数十名从业者，他们都是这项试图导致LLMS失败的新奇工作的贡献者。我们专注于定义LLM红色团队，揭示执行该活动的动机和目标，以及表征人们在攻击LLM时使用的策略等研究问题。根据数据，LLM红色团队被定义为一种寻求极限的非恶意手动活动，高度依赖于团队努力和炼金术士的心态。这是出于好奇心和乐趣的高度内在动机，在某种程度上也是出于对部署LLMS的各种危害的担忧。我们确定了攻击LLMS的12种策略和35种不同技术的分类。这些发现是关于人们如何以及为什么攻击大型语言模型的一个全面的扎根理论：LLM红色团队。



## **45. FlexLLM: Exploring LLM Customization for Moving Target Defense on Black-Box LLMs Against Jailbreak Attacks**

FlexLLM：探索LLM定制，以在黑匣子LLM上移动目标防御以防止越狱攻击 cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07672v1) [paper-pdf](http://arxiv.org/pdf/2412.07672v1)

**Authors**: Bocheng Chen, Hanqing Guo, Qiben Yan

**Abstract**: Defense in large language models (LLMs) is crucial to counter the numerous attackers exploiting these systems to generate harmful content through manipulated prompts, known as jailbreak attacks. Although many defense strategies have been proposed, they often require access to the model's internal structure or need additional training, which is impractical for service providers using LLM APIs, such as OpenAI APIs or Claude APIs. In this paper, we propose a moving target defense approach that alters decoding hyperparameters to enhance model robustness against various jailbreak attacks. Our approach does not require access to the model's internal structure and incurs no additional training costs. The proposed defense includes two key components: (1) optimizing the decoding strategy by identifying and adjusting decoding hyperparameters that influence token generation probabilities, and (2) transforming the decoding hyperparameters and model system prompts into dynamic targets, which are continuously altered during each runtime. By continuously modifying decoding strategies and prompts, the defense effectively mitigates the existing attacks. Our results demonstrate that our defense is the most effective against jailbreak attacks in three of the models tested when using LLMs as black-box APIs. Moreover, our defense offers lower inference costs and maintains comparable response quality, making it a potential layer of protection when used alongside other defense methods.

摘要: 大型语言模型(LLMS)中的防御对于对抗大量攻击者至关重要，这些攻击者利用这些系统通过被操纵的提示生成有害内容，即所谓的越狱攻击。虽然已经提出了许多防御策略，但它们往往需要访问模型的内部结构或需要额外的培训，这对于使用LLM API的服务提供商来说是不切实际的，例如OpenAI API或Claude API。在本文中，我们提出了一种移动目标防御方法，通过改变解码超参数来增强模型对各种越狱攻击的稳健性。我们的方法不需要访问模型的内部结构，也不会产生额外的培训成本。该防御包括两个关键部分：(1)通过识别和调整影响令牌生成概率的解码超参数来优化解码策略；(2)将解码超参数和模型系统提示转换为动态目标，这些目标在每次运行时不断变化。通过不断修改解码策略和提示，有效缓解了现有攻击。我们的结果表明，当使用LLM作为黑盒API时，在测试的三个模型中，我们的防御是最有效的越狱攻击。此外，我们的防御提供了更低的推理成本，并保持了相当的响应质量，使其在与其他防御方法一起使用时成为潜在的保护层。



## **46. Look Before You Leap: Enhancing Attention and Vigilance Regarding Harmful Content with GuidelineLLM**

三思而后行：通过GuidelineLLM提高对有害内容的关注和警惕 cs.CL

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.10423v1) [paper-pdf](http://arxiv.org/pdf/2412.10423v1)

**Authors**: Shaoqing Zhang, Zhuosheng Zhang, Kehai Chen, Rongxiang Weng, Muyun Yang, Tiejun Zhao, Min Zhang

**Abstract**: Despite being empowered with alignment mechanisms, large language models (LLMs) are increasingly vulnerable to emerging jailbreak attacks that can compromise their alignment mechanisms. This vulnerability poses significant risks to the real-world applications. Existing work faces challenges in both training efficiency and generalization capabilities (i.e., Reinforcement Learning from Human Feedback and Red-Teaming). Developing effective strategies to enable LLMs to resist continuously evolving jailbreak attempts represents a significant challenge. To address this challenge, we propose a novel defensive paradigm called GuidelineLLM, which assists LLMs in recognizing queries that may have harmful content. Before LLMs respond to a query, GuidelineLLM first identifies potential risks associated with the query, summarizes these risks into guideline suggestions, and then feeds these guidelines to the responding LLMs. Importantly, our approach eliminates the necessity for additional safety fine-tuning of the LLMs themselves; only the GuidelineLLM requires fine-tuning. This characteristic enhances the general applicability of GuidelineLLM across various LLMs. Experimental results demonstrate that GuidelineLLM can significantly reduce the attack success rate (ASR) against the LLMs (an average reduction of 34.17\% ASR) while maintaining the helpfulness of the LLMs in handling benign queries. Code is available at https://github.com/sqzhang-lazy/GuidelineLLM.

摘要: 该漏洞给现实世界中的应用程序带来了重大风险。现有的工作在训练效率和泛化能力(即来自人的反馈的强化学习和红队)方面都面临挑战。制定有效的战略，使低收入国家能够抵抗不断演变的越狱企图，这是一项重大挑战。为了应对这一挑战，我们提出了一种名为GuidelineLLM的新的防御范例，它帮助LLM识别可能包含有害内容的查询。在LLM响应查询之前，GuidelineLLM首先识别与查询相关的潜在风险，将这些风险汇总到指南建议中，然后将这些指南反馈给响应的LLM。重要的是，我们的方法消除了对LLM本身进行额外安全微调的必要性；只有GuidelineLLM需要微调。这一特性增强了GuidelineLLM在各种LLM中的普遍适用性。实验结果表明，GuidelineLLM能够显著降低对LLMS的攻击成功率(ASR)(平均降低34.17ASR)，同时保持LLMS对良性查询的帮助。代码可在https://github.com/sqzhang-lazy/GuidelineLLM.上找到



## **47. SQL Injection Jailbreak: a structural disaster of large language models**

SQL注入越狱：大型语言模型的结构性灾难 cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2411.01565v3) [paper-pdf](http://arxiv.org/pdf/2411.01565v3)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: In recent years, the rapid development of large language models (LLMs) has brought new vitality into various domains, generating substantial social and economic benefits. However, this swift advancement has also introduced new security vulnerabilities. Jailbreaking, a form of attack that induces LLMs to produce harmful content through carefully crafted prompts, presents a significant challenge to the safe and trustworthy development of LLMs. Previous jailbreak methods primarily exploited the internal properties or capabilities of LLMs, such as optimization-based jailbreak approaches and methods that leveraged the model's context-learning abilities. In this paper, we introduce a novel jailbreak method, SQL Injection Jailbreak (SIJ), which targets the external properties of LLMs, specifically, the way LLMs construct input prompts. By injecting jailbreak information into user prompts, SIJ successfully induces the model to output harmful content. Our SIJ method achieves near 100\% attack success rates on five well-known open-source LLMs on the AdvBench, while incurring lower time costs compared to previous methods. More importantly, SIJ is the first method to exploit the external properties of LLMs for jailbreak attacks and exposes a new vulnerability in LLMs that urgently requires mitigation. To address this, we propose a simple defense method called Self-Reminder-Key to counter SIJ and demonstrate its effectiveness through experimental results. Our code is available at \href{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}.

摘要: 近年来，大型语言模型的快速发展为各个领域带来了新的活力，产生了可观的社会效益和经济效益。然而，这种快速发展也带来了新的安全漏洞。越狱是一种攻击形式，通过精心制作的提示诱使LLM产生有害内容，对LLM的安全和可信开发构成了重大挑战。以前的越狱方法主要利用LLMS的内部属性或功能，例如基于优化的越狱方法和利用模型的上下文学习能力的方法。本文介绍了一种新的越狱方法--SQL注入越狱(SIJ)，它针对LLMS的外部属性，特别是LLMS构造输入提示的方式。通过在用户提示中注入越狱信息，SIJ成功地诱导该模型输出有害内容。与以前的方法相比，我们的SIJ方法在AdvBch上的五个著名的开源LLM上获得了近100%的攻击成功率，同时产生了更低的时间成本。更重要的是，SIJ是第一个利用LLMS的外部属性进行越狱攻击的方法，并暴露了LLMS中一个迫切需要缓解的新漏洞。针对这一问题，我们提出了一种简单的防御方法，称为自我提醒密钥来对抗SIJ，并通过实验结果证明了其有效性。我们的代码可以在\href{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}.上找到



## **48. MobileSafetyBench: Evaluating Safety of Autonomous Agents in Mobile Device Control**

MobileSafetyBench：评估移动终端控制中自治代理的安全性 cs.LG

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2410.17520v2) [paper-pdf](http://arxiv.org/pdf/2410.17520v2)

**Authors**: Juyong Lee, Dongyoon Hahm, June Suk Choi, W. Bradley Knox, Kimin Lee

**Abstract**: Autonomous agents powered by large language models (LLMs) show promising potential in assistive tasks across various domains, including mobile device control. As these agents interact directly with personal information and device settings, ensuring their safe and reliable behavior is crucial to prevent undesirable outcomes. However, no benchmark exists for standardized evaluation of the safety of mobile device-control agents. In this work, we introduce MobileSafetyBench, a benchmark designed to evaluate the safety of device-control agents within a realistic mobile environment based on Android emulators. We develop a diverse set of tasks involving interactions with various mobile applications, including messaging and banking applications, challenging agents with managing risks encompassing misuse and negative side effects. These tasks include tests to evaluate the safety of agents in daily scenarios as well as their robustness against indirect prompt injection attacks. Our experiments demonstrate that baseline agents, based on state-of-the-art LLMs, often fail to effectively prevent harm while performing the tasks. To mitigate these safety concerns, we propose a prompting method that encourages agents to prioritize safety considerations. While this method shows promise in promoting safer behaviors, there is still considerable room for improvement to fully earn user trust. This highlights the urgent need for continued research to develop more robust safety mechanisms in mobile environments. We open-source our benchmark at: https://mobilesafetybench.github.io/.

摘要: 由大语言模型(LLM)驱动的自主代理在包括移动设备控制在内的各个领域的辅助任务中显示出巨大的潜力。由于这些代理直接与个人信息和设备设置交互，因此确保其安全可靠的行为对于防止不良结果至关重要。然而，移动设备控制代理的安全标准化评估尚不存在基准。在这项工作中，我们引入了MobileSafetyBch，这是一个旨在基于Android模拟器在现实移动环境中评估设备控制代理的安全性的基准测试。我们开发了一套多样化的任务，涉及与各种移动应用程序的交互，包括消息传递和银行应用程序，挑战代理商管理包括滥用和负面副作用在内的风险。这些任务包括测试，以评估代理在日常场景中的安全性，以及它们对间接快速注入攻击的稳健性。我们的实验表明，基于最先进的LLM的基线代理在执行任务时往往无法有效地防止伤害。为了缓解这些安全问题，我们提出了一种提示方法，鼓励工程师优先考虑安全因素。虽然这种方法在促进更安全的行为方面表现出了希望，但要充分赢得用户信任，仍有相当大的改进空间。这突显了迫切需要继续研究，以在移动环境中开发更强大的安全机制。我们在https://mobilesafetybench.github.io/.上开放了我们的基准测试



## **49. Na'vi or Knave: Jailbreaking Language Models via Metaphorical Avatars**

纳威人或恶棍：通过隐喻化身越狱语言模型 cs.CL

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.12145v1) [paper-pdf](http://arxiv.org/pdf/2412.12145v1)

**Authors**: Yu Yan, Sheng Sun, Junqi Tong, Min Liu, Qi Li

**Abstract**: Metaphor serves as an implicit approach to convey information, while enabling the generalized comprehension of complex subjects. However, metaphor can potentially be exploited to bypass the safety alignment mechanisms of Large Language Models (LLMs), leading to the theft of harmful knowledge. In our study, we introduce a novel attack framework that exploits the imaginative capacity of LLMs to achieve jailbreaking, the J\underline{\textbf{A}}ilbreak \underline{\textbf{V}}ia \underline{\textbf{A}}dversarial Me\underline{\textbf{TA}} -pho\underline{\textbf{R}} (\textit{AVATAR}). Specifically, to elicit the harmful response, AVATAR extracts harmful entities from a given harmful target and maps them to innocuous adversarial entities based on LLM's imagination. Then, according to these metaphors, the harmful target is nested within human-like interaction for jailbreaking adaptively. Experimental results demonstrate that AVATAR can effectively and transferablly jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. Our study exposes a security risk in LLMs from their endogenous imaginative capabilities. Furthermore, the analytical study reveals the vulnerability of LLM to adversarial metaphors and the necessity of developing defense methods against jailbreaking caused by the adversarial metaphor. \textcolor{orange}{ \textbf{Warning: This paper contains potentially harmful content from LLMs.}}

摘要: 隐喻是一种隐含的传达信息的方法，同时也使人们能够对复杂的主题进行广义的理解。然而，隐喻可能被利用来绕过大型语言模型(LLM)的安全对齐机制，从而导致有害知识的窃取。在我们的研究中，我们提出了一种新的攻击框架，该框架利用LLMS的想象能力来实现越狱，即J下划线{\extbf{A}}ilBreak\下划线{\extbf{A}}ia\下划线{\extbf{A}}dversarial Me\下划线{\extbf{TA}}-pho\下划线{\extbf{R}}(\textit{阿凡达})。具体地说，为了引发有害反应，阿凡达从给定的有害目标中提取有害实体，并基于LLM的想象力将它们映射到无害的对抗性实体。然后，根据这些隐喻，有害的目标嵌套在类似人类的互动中，以适应越狱。实验结果表明，阿凡达可以有效地、可转移地越狱LLM，并在多个高级LLM上获得最先进的攻击成功率。我们的研究揭示了LLMS的安全风险来自其内生的想象力。此外，分析研究揭示了LLM对对抗性隐喻的脆弱性，以及开发针对对抗性隐喻导致的越狱的防御方法的必要性。\extCOLOR{橙色}{\extbf{警告：此纸张包含来自LLMS的潜在有害内容。}}



## **50. PrisonBreak: Jailbreaking Large Language Models with Fewer Than Twenty-Five Targeted Bit-flips**

Prison Break：越狱大型语言模型，目标位翻转少于25个 cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07192v1) [paper-pdf](http://arxiv.org/pdf/2412.07192v1)

**Authors**: Zachary Coalson, Jeonghyun Woo, Shiyang Chen, Yu Sun, Lishan Yang, Prashant Nair, Bo Fang, Sanghyun Hong

**Abstract**: We introduce a new class of attacks on commercial-scale (human-aligned) language models that induce jailbreaking through targeted bitwise corruptions in model parameters. Our adversary can jailbreak billion-parameter language models with fewer than 25 bit-flips in all cases$-$and as few as 5 in some$-$using up to 40$\times$ less bit-flips than existing attacks on computer vision models at least 100$\times$ smaller. Unlike prompt-based jailbreaks, our attack renders these models in memory 'uncensored' at runtime, allowing them to generate harmful responses without any input modifications. Our attack algorithm efficiently identifies target bits to flip, offering up to 20$\times$ more computational efficiency than previous methods. This makes it practical for language models with billions of parameters. We show an end-to-end exploitation of our attack using software-induced fault injection, Rowhammer (RH). Our work examines 56 DRAM RH profiles from DDR4 and LPDDR4X devices with different RH vulnerabilities. We show that our attack can reliably induce jailbreaking in systems similar to those affected by prior bit-flip attacks. Moreover, our approach remains effective even against highly RH-secure systems (e.g., 46$\times$ more secure than previously tested systems). Our analyses further reveal that: (1) models with less post-training alignment require fewer bit flips to jailbreak; (2) certain model components, such as value projection layers, are substantially more vulnerable than others; and (3) our method is mechanistically different than existing jailbreaks. Our findings highlight a pressing, practical threat to the language model ecosystem and underscore the need for research to protect these models from bit-flip attacks.

摘要: 我们在商业规模(人类对齐的)语言模型上引入了一类新的攻击，这些攻击通过模型参数中有针对性的逐位破坏来诱导越狱。我们的对手可以用不到25个比特翻转的语言模型越狱，所有情况下都不到25个比特翻转，在一些$-$中只有5个，使用多达40个比特翻转，比对计算机视觉模型的现有攻击少至少100$\×$。与基于提示的越狱不同，我们的攻击在运行时将这些模型呈现在内存中，不受审查，允许它们在不修改任何输入的情况下生成有害的响应。我们的攻击算法有效地识别要翻转的目标比特，比以前的方法提供了高达20美元\倍的计算效率。这使得它适用于具有数十亿个参数的语言模型。我们使用软件诱导的故障注入Rowhammer(RH)展示了对我们的攻击的端到端攻击。我们的工作检查了来自具有不同RH漏洞的DDR4和LPDDR4X设备的56个DRAM RH配置文件。我们证明了我们的攻击可以可靠地在类似于先前受比特翻转攻击影响的系统中诱导越狱。此外，我们的方法即使对高度RH安全的系统也是有效的(例如，比之前测试的系统安全46美元\倍)。我们的分析进一步表明：(1)训练后对齐较少的模型需要较少的比特翻转越狱；(2)某些模型组件，如值投影层，比其他组件更容易受到攻击；(3)我们的方法与现有的越狱方法在机械上不同。我们的发现突显了语言模型生态系统面临的紧迫、实际的威胁，并强调了研究保护这些模型免受比特翻转攻击的必要性。



