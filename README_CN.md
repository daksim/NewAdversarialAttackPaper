# Latest Adversarial Attack Papers
**update at 2024-10-31 09:51:21**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

DistriBlock：通过利用输出分布的特征来识别对抗性音频样本 cs.SD

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2305.17000v7) [paper-pdf](http://arxiv.org/pdf/2305.17000v7)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.

摘要: 敌意攻击可以误导自动语音识别(ASR)系统预测任意目标文本，从而构成明显的安全威胁。为了防止此类攻击，我们提出了DistriBlock，这是一种适用于任何ASR系统的有效检测策略，它预测每个时间步输出令牌上的概率分布。我们测量了该分布的一组特征：输出概率的中位数、最大值和最小值，分布的熵，以及关于后续时间步分布的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性数据和恶意数据观察到的特征，我们应用二进制分类器，包括简单的基于阈值的分类、这种分类器的集成和神经网络。通过对不同的ASR系统和语言数据集的广泛分析，我们证明了该方法的最高性能，在干净和有噪声的数据下，接收器操作特征曲线下的平均面积分别为99%和97%。为了评估我们方法的健壮性，我们证明了可以绕过DistriBlock的自适应攻击示例的噪声要大得多，这使得它们更容易通过过滤来检测，并为保持系统的健壮性创造了另一种途径。



## **2. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

卡西姆·分歧：通过对抗性防御的扩散模型来启发性解纠缠 cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.23091v1) [paper-pdf](http://arxiv.org/pdf/2410.23091v1)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark).

摘要: 尽管不断努力保护神经分类器免受对手攻击，但它们仍然很脆弱，特别是面对看不见的攻击。相比之下，人类很难被微妙的操纵所欺骗，因为我们只根据基本因素做出判断。受到这一观察的启发，我们试图用基本的标签原因因素来建模标签生成，并结合标签非原因因素来辅助数据生成。对于一个对抗性的例子，我们的目标是将扰动区分为非致因因素，并仅基于标签致因因素进行预测。具体地说，我们提出了一个偶然扩散模型(CausalDiff)，该模型使扩散模型适用于条件数据生成，并通过向一个新的偶然信息瓶颈目标学习来区分这两种类型的偶然因素。经验上，CausalDiff在各种隐形攻击上的表现明显优于最先进的防御方法，在CIFAR-10上获得了86.39%(+4.01%)的平均健壮性，在CIFAR-100上获得了56.25%(+3.13%)的健壮性，在GTSRB(德国交通标志识别基准)上实现了82.62%(+4.93%)的平均健壮性。



## **3. Robustifying automatic speech recognition by extracting slowly varying features**

通过提取缓慢变化的特征来增强自动语音识别 eess.AS

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2112.07400v2) [paper-pdf](http://arxiv.org/pdf/2112.07400v2)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: In the past few years, it has been shown that deep learning systems are highly vulnerable under attacks with adversarial examples. Neural-network-based automatic speech recognition (ASR) systems are no exception. Targeted and untargeted attacks can modify an audio input signal in such a way that humans still recognise the same words, while ASR systems are steered to predict a different transcription. In this paper, we propose a defense mechanism against targeted adversarial attacks consisting in removing fast-changing features from the audio signals, either by applying slow feature analysis, a low-pass filter, or both, before feeding the input to the ASR system. We perform an empirical analysis of hybrid ASR models trained on data pre-processed in such a way. While the resulting models perform quite well on benign data, they are significantly more robust against targeted adversarial attacks: Our final, proposed model shows a performance on clean data similar to the baseline model, while being more than four times more robust.

摘要: 在过去的几年里，深度学习系统被证明在敌意攻击下是非常脆弱的。基于神经网络的自动语音识别(ASR)系统也不例外。定向和非定向攻击可以修改音频输入信号，使人类仍能识别相同的单词，而ASR系统则被引导预测不同的转录。在本文中，我们提出了一种针对目标攻击的防御机制，即在将输入输入到ASR系统之前，通过慢速特征分析、低通滤波或两者结合的方法，从音频信号中去除快速变化的特征。我们对以这种方式处理的数据训练的混合ASR模型进行了实证分析。虽然得到的模型在良性数据上表现得相当好，但它们对目标对手攻击的健壮性要强得多：我们最终提出的模型在干净数据上的性能与基准模型相似，但健壮性要高四倍以上。



## **4. Are Your Models Still Fair? Fairness Attacks on Graph Neural Networks via Node Injections**

你的模型仍然公平吗？通过节点注入对图神经网络的公平性攻击 cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2406.03052v2) [paper-pdf](http://arxiv.org/pdf/2406.03052v2)

**Authors**: Zihan Luo, Hong Huang, Yongkang Zhou, Jiping Zhang, Nuo Chen, Hai Jin

**Abstract**: Despite the remarkable capabilities demonstrated by Graph Neural Networks (GNNs) in graph-related tasks, recent research has revealed the fairness vulnerabilities in GNNs when facing malicious adversarial attacks. However, all existing fairness attacks require manipulating the connectivity between existing nodes, which may be prohibited in reality. To this end, we introduce a Node Injection-based Fairness Attack (NIFA), exploring the vulnerabilities of GNN fairness in such a more realistic setting. In detail, NIFA first designs two insightful principles for node injection operations, namely the uncertainty-maximization principle and homophily-increase principle, and then optimizes injected nodes' feature matrix to further ensure the effectiveness of fairness attacks. Comprehensive experiments on three real-world datasets consistently demonstrate that NIFA can significantly undermine the fairness of mainstream GNNs, even including fairness-aware GNNs, by injecting merely 1% of nodes. We sincerely hope that our work can stimulate increasing attention from researchers on the vulnerability of GNN fairness, and encourage the development of corresponding defense mechanisms. Our code and data are released at: https://github.com/CGCL-codes/NIFA.

摘要: 尽管图神经网络(GNN)在与图相关的任务中表现出了卓越的能力，但最近的研究揭示了GNN在面对恶意攻击时的公平性漏洞。然而，所有现有的公平攻击都需要操纵现有节点之间的连通性，这在现实中可能是被禁止的。为此，我们引入了一种基于节点注入的公平攻击(NIFA)，探讨了GNN公平性在这样一个更现实的环境下的脆弱性。具体而言，NIFA首先为节点注入操作设计了两个有洞察力的原则，即不确定性最大化原则和同质性增加原则，然后对注入节点的特征矩阵进行优化，进一步保证了公平攻击的有效性。在三个真实数据集上的综合实验一致表明，NIFA只注入1%的节点，就可以显著破坏主流GNN的公平性，即使包括公平感知的GNN。我们真诚地希望我们的工作能够引起研究人员对GNN公平性脆弱性的越来越多的关注，并鼓励开发相应的防御机制。我们的代码和数据发布在：https://github.com/CGCL-codes/NIFA.



## **5. Effective and Efficient Adversarial Detection for Vision-Language Models via A Single Vector**

通过单个载体对视觉语言模型进行有效且高效的对抗检测 cs.CV

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22888v1) [paper-pdf](http://arxiv.org/pdf/2410.22888v1)

**Authors**: Youcheng Huang, Fengbin Zhu, Jingkun Tang, Pan Zhou, Wenqiang Lei, Jiancheng Lv, Tat-Seng Chua

**Abstract**: Visual Language Models (VLMs) are vulnerable to adversarial attacks, especially those from adversarial images, which is however under-explored in literature. To facilitate research on this critical safety problem, we first construct a new laRge-scale Adervsarial images dataset with Diverse hArmful Responses (RADAR), given that existing datasets are either small-scale or only contain limited types of harmful responses. With the new RADAR dataset, we further develop a novel and effective iN-time Embedding-based AdveRSarial Image DEtection (NEARSIDE) method, which exploits a single vector that distilled from the hidden states of VLMs, which we call the attacking direction, to achieve the detection of adversarial images against benign ones in the input. Extensive experiments with two victim VLMs, LLaVA and MiniGPT-4, well demonstrate the effectiveness, efficiency, and cross-model transferrability of our proposed method. Our code is available at https://github.com/mob-scu/RADAR-NEARSIDE

摘要: 视觉语言模型（VLM）很容易受到对抗性攻击，尤其是来自对抗性图像的攻击，但文献中对此尚未充分探讨。为了促进对这个关键安全问题的研究，我们首先构建一个具有多样性干扰响应（RADART）的新的大规模Adervsarial图像数据集，因为现有数据集要么小规模，要么仅包含有限类型的有害反应。利用新的雷达数据集，我们进一步开发了一种新颖且有效的基于iN时间嵌入的AdveRSarial Image Detect（NEARSIDE）方法，该方法利用从VLM的隐藏状态（我们称之为攻击方向）中提取的单个载体，以实现针对输入中良性图像的对抗图像的检测。对两个受害VLM（LLaVA和MiniGPT-4）进行的大量实验很好地证明了我们提出的方法的有效性、效率和跨模型可移植性。我们的代码可在https://github.com/mob-scu/RADAR-NEARSIDE上获取



## **6. Stealing User Prompts from Mixture of Experts**

从专家混合处窃取用户预算 cs.CR

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22884v1) [paper-pdf](http://arxiv.org/pdf/2410.22884v1)

**Authors**: Itay Yona, Ilia Shumailov, Jamie Hayes, Nicholas Carlini

**Abstract**: Mixture-of-Experts (MoE) models improve the efficiency and scalability of dense language models by routing each token to a small number of experts in each layer. In this paper, we show how an adversary that can arrange for their queries to appear in the same batch of examples as a victim's queries can exploit Expert-Choice-Routing to fully disclose a victim's prompt. We successfully demonstrate the effectiveness of this attack on a two-layer Mixtral model, exploiting the tie-handling behavior of the torch.topk CUDA implementation. Our results show that we can extract the entire prompt using $O({VM}^2)$ queries (with vocabulary size $V$ and prompt length $M$) or 100 queries on average per token in the setting we consider. This is the first attack to exploit architectural flaws for the purpose of extracting user prompts, introducing a new class of LLM vulnerabilities.

摘要: 专家混合（MoE）模型通过将每个令牌路由到每层中的少数专家来提高密集语言模型的效率和可扩展性。在本文中，我们展示了可以安排其查询与受害者查询出现在同一批示例中的对手如何利用Expert-Choice-Routing来完全披露受害者的提示。我们利用torch.topk CUDA实现的领带处理行为，在两层Mixtral模型上成功证明了这种攻击的有效性。我们的结果表明，我们可以使用$O（{VM}'#39;#39;#39; s）$查询（词汇量大小为$V$，提示长度为$M$）或在我们考虑的设置中每个令牌平均提取100个查询来提取整个提示。这是第一次利用架构缺陷来提取用户提示的攻击，从而引入了一类新的LLM漏洞。



## **7. Understanding and Improving Adversarial Collaborative Filtering for Robust Recommendation**

了解和改进对抗性协作过滤以实现稳健推荐 cs.IR

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22844v1) [paper-pdf](http://arxiv.org/pdf/2410.22844v1)

**Authors**: Kaike Zhang, Qi Cao, Yunfan Wu, Fei Sun, Huawei Shen, Xueqi Cheng

**Abstract**: Adversarial Collaborative Filtering (ACF), which typically applies adversarial perturbations at user and item embeddings through adversarial training, is widely recognized as an effective strategy for enhancing the robustness of Collaborative Filtering (CF) recommender systems against poisoning attacks. Besides, numerous studies have empirically shown that ACF can also improve recommendation performance compared to traditional CF. Despite these empirical successes, the theoretical understanding of ACF's effectiveness in terms of both performance and robustness remains unclear. To bridge this gap, in this paper, we first theoretically show that ACF can achieve a lower recommendation error compared to traditional CF with the same training epochs in both clean and poisoned data contexts. Furthermore, by establishing bounds for reductions in recommendation error during ACF's optimization process, we find that applying personalized magnitudes of perturbation for different users based on their embedding scales can further improve ACF's effectiveness. Building on these theoretical understandings, we propose Personalized Magnitude Adversarial Collaborative Filtering (PamaCF). Extensive experiments demonstrate that PamaCF effectively defends against various types of poisoning attacks while significantly enhancing recommendation performance.

摘要: 对抗性协同过滤(ACF)通过对抗性训练将对抗性扰动应用于用户和项目嵌入，被广泛认为是提高协同过滤推荐系统对中毒攻击的稳健性的有效策略。此外，大量研究表明，与传统的推荐算法相比，自适应过滤算法也能提高推荐性能。尽管取得了这些经验上的成功，但关于ACF在性能和稳健性方面的有效性的理论理解仍然不清楚。为了弥补这一差距，在本文中，我们首先从理论上证明，在干净和有毒的数据环境下，与相同训练周期的传统CF相比，ACF可以获得更低的推荐误差。此外，通过建立ACF优化过程中推荐误差减少的界限，我们发现根据不同用户的嵌入尺度对不同用户应用个性化的扰动幅度可以进一步提高ACF的有效性。在这些理论理解的基础上，我们提出了个性化幅度对抗协同过滤(PamaCF)。大量实验表明，PamaCF在有效防御各种类型的中毒攻击的同时，显著提高了推荐性能。



## **8. One Prompt to Verify Your Models: Black-Box Text-to-Image Models Verification via Non-Transferable Adversarial Attacks**

验证模型的一个提示：通过不可传输对抗性攻击进行黑匣子文本到图像模型验证 cs.CV

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22725v1) [paper-pdf](http://arxiv.org/pdf/2410.22725v1)

**Authors**: Ji Guo, Wenbo Jiang, Rui Zhang, Guoming Lu, Hongwei Li, Weiren Wu

**Abstract**: Recently, the success of Text-to-Image (T2I) models has led to the rise of numerous third-party platforms, which claim to provide cheaper API services and more flexibility in model options. However, this also raises a new security concern: Are these third-party services truly offering the models they claim? To address this problem, we propose the first T2I model verification method named Text-to-Image Model Verification via Non-Transferable Adversarial Attacks (TVN). The non-transferability of adversarial examples means that these examples are only effective on a target model and ineffective on other models, thereby allowing for the verification of the target model. TVN utilizes the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to optimize the cosine similarity of a prompt's text encoding, generating non-transferable adversarial prompts. By calculating the CLIP-text scores between the non-transferable adversarial prompts without perturbations and the images, we can verify if the model matches the claimed target model, based on a 3-sigma threshold. The experiments showed that TVN performed well in both closed-set and open-set scenarios, achieving a verification accuracy of over 90\%. Moreover, the adversarial prompts generated by TVN significantly reduced the CLIP-text scores of the target model, while having little effect on other models.

摘要: 最近，文本到图像(T2I)模式的成功导致了无数第三方平台的崛起，这些平台声称提供更便宜的API服务和更灵活的模式选择。然而，这也引发了一个新的安全问题：这些第三方服务是否真的提供了它们声称的模式？针对这一问题，我们提出了第一种T2I模型验证方法--基于不可转移攻击的文本到图像模型验证方法(TVN)。对抗性实例的不可转移性意味着这些实例仅对目标模型有效，而对其他模型无效，从而允许对目标模型进行验证。TVN使用非支配排序遗传算法II(NSGA-II)来优化提示文本编码的余弦相似度，生成不可转移的对抗性提示。通过计算不可转移的敌意提示与图像之间的剪贴文本分数，我们可以基于3-sigma阈值来验证该模型是否与所声称的目标模型匹配。实验表明，TVN在闭集和开集场景下都表现良好，验证准确率达到90%以上。此外，TVN生成的对抗性提示显著降低了目标模型的片段文本分数，而对其他模型影响不大。



## **9. Geometry Cloak: Preventing TGS-based 3D Reconstruction from Copyrighted Images**

几何斗篷：防止受版权保护的图像进行基于TG的3D重建 cs.CV

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22705v1) [paper-pdf](http://arxiv.org/pdf/2410.22705v1)

**Authors**: Qi Song, Ziyuan Luo, Ka Chun Cheung, Simon See, Renjie Wan

**Abstract**: Single-view 3D reconstruction methods like Triplane Gaussian Splatting (TGS) have enabled high-quality 3D model generation from just a single image input within seconds. However, this capability raises concerns about potential misuse, where malicious users could exploit TGS to create unauthorized 3D models from copyrighted images. To prevent such infringement, we propose a novel image protection approach that embeds invisible geometry perturbations, termed "geometry cloaks", into images before supplying them to TGS. These carefully crafted perturbations encode a customized message that is revealed when TGS attempts 3D reconstructions of the cloaked image. Unlike conventional adversarial attacks that simply degrade output quality, our method forces TGS to fail the 3D reconstruction in a specific way - by generating an identifiable customized pattern that acts as a watermark. This watermark allows copyright holders to assert ownership over any attempted 3D reconstructions made from their protected images. Extensive experiments have verified the effectiveness of our geometry cloak. Our project is available at https://qsong2001.github.io/geometry_cloak.

摘要: 像三平面高斯飞溅(TGS)这样的单视图3D重建方法能够在几秒钟内从单一图像输入生成高质量的3D模型。然而，这一功能引发了人们对潜在滥用的担忧，恶意用户可能会利用TGS从受版权保护的图像创建未经授权的3D模型。为了防止这种侵权行为，我们提出了一种新的图像保护方法，该方法在将不可见的几何扰动(称为几何斗篷)嵌入到图像中，然后将其提供给TGS。这些精心制作的扰动编码了一条定制的消息，当TGS试图对被遮盖的图像进行3D重建时，该消息会被揭示出来。与简单地降低输出质量的传统对抗性攻击不同，我们的方法迫使TGS以特定的方式失败3D重建-通过生成可识别的定制图案作为水印。该水印允许版权所有者主张对从其受保护的图像进行的任何3D重建尝试的所有权。广泛的实验已经验证了我们几何斗篷的有效性。我们的项目可在https://qsong2001.github.io/geometry_cloak.上查看



## **10. Backdoor Attack Against Vision Transformers via Attention Gradient-Based Image Erosion**

通过基于注意力的图像侵蚀对视觉变形者进行后门攻击 cs.CV

Accepted by IEEE GLOBECOM 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22678v1) [paper-pdf](http://arxiv.org/pdf/2410.22678v1)

**Authors**: Ji Guo, Hongwei Li, Wenbo Jiang, Guoming Lu

**Abstract**: Vision Transformers (ViTs) have outperformed traditional Convolutional Neural Networks (CNN) across various computer vision tasks. However, akin to CNN, ViTs are vulnerable to backdoor attacks, where the adversary embeds the backdoor into the victim model, causing it to make wrong predictions about testing samples containing a specific trigger. Existing backdoor attacks against ViTs have the limitation of failing to strike an optimal balance between attack stealthiness and attack effectiveness.   In this work, we propose an Attention Gradient-based Erosion Backdoor (AGEB) targeted at ViTs. Considering the attention mechanism of ViTs, AGEB selectively erodes pixels in areas of maximal attention gradient, embedding a covert backdoor trigger. Unlike previous backdoor attacks against ViTs, AGEB achieves an optimal balance between attack stealthiness and attack effectiveness, ensuring the trigger remains invisible to human detection while preserving the model's accuracy on clean samples. Extensive experimental evaluations across various ViT architectures and datasets confirm the effectiveness of AGEB, achieving a remarkable Attack Success Rate (ASR) without diminishing Clean Data Accuracy (CDA). Furthermore, the stealthiness of AGEB is rigorously validated, demonstrating minimal visual discrepancies between the clean and the triggered images.

摘要: 视觉转换器(VITS)在各种计算机视觉任务中的表现优于传统的卷积神经网络(CNN)。然而，与CNN类似，VITS容易受到后门攻击，即对手将后门嵌入受害者模型，导致其对包含特定触发因素的测试样本做出错误预测。现有的针对VITS的后门攻击存在未能在攻击隐蔽性和攻击有效性之间取得最佳平衡的局限性。在这项工作中，我们提出了一个针对VITS的基于注意力梯度的侵蚀后门(AGEB)。考虑到VITS的注意机制，AGEB通过嵌入一个隐蔽的后门触发器，选择性地侵蚀注意力梯度最大的区域的像素。与以前针对VITS的后门攻击不同，AGEB在攻击隐蔽性和攻击有效性之间实现了最佳平衡，确保触发器保持人类检测不到，同时保持模型对干净样本的准确性。对各种VIT架构和数据集进行的广泛实验评估证实了AGEB的有效性，在不降低清洁数据准确性(CDA)的情况下实现了显著的攻击成功率(ASR)。此外，AGEB的隐蔽性得到了严格的验证，显示了干净图像和触发图像之间最小的视觉差异。



## **11. Automated Trustworthiness Oracle Generation for Machine Learning Text Classifiers**

用于机器学习文本分类器的自动可信度Oracle生成 cs.SE

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22663v1) [paper-pdf](http://arxiv.org/pdf/2410.22663v1)

**Authors**: Lam Nguyen Tung, Steven Cho, Xiaoning Du, Neelofar Neelofar, Valerio Terragni, Stefano Ruberto, Aldeida Aleti

**Abstract**: Machine learning (ML) for text classification has been widely used in various domains, such as toxicity detection, chatbot consulting, and review analysis. These applications can significantly impact ethics, economics, and human behavior, raising serious concerns about trusting ML decisions. Several studies indicate that traditional metrics, such as model confidence and accuracy, are insufficient to build human trust in ML models. These models often learn spurious correlations during training and predict based on them during inference. In the real world, where such correlations are absent, their performance can deteriorate significantly. To avoid this, a common practice is to test whether predictions are reasonable. Along with this, a challenge known as the trustworthiness oracle problem has been introduced. Due to the lack of automated trustworthiness oracles, the assessment requires manual validation of the decision process disclosed by explanation methods, which is time-consuming and not scalable. We propose TOKI, the first automated trustworthiness oracle generation method for text classifiers, which automatically checks whether the prediction-contributing words are related to the predicted class using explanation methods and word embeddings. To demonstrate its practical usefulness, we introduce a novel adversarial attack method targeting trustworthiness issues identified by TOKI. We compare TOKI with a naive baseline based solely on model confidence using human-created ground truths of 6,000 predictions. We also compare TOKI-guided adversarial attack method with A2T, a SOTA adversarial attack method. Results show that relying on prediction uncertainty cannot distinguish between trustworthy and untrustworthy predictions, TOKI achieves 142% higher accuracy than the naive baseline, and TOKI-guided adversarial attack method is more effective with fewer perturbations than A2T.

摘要: 机器学习用于文本分类已被广泛应用于毒性检测、聊天机器人咨询、评论分析等领域。这些应用程序可能会对伦理、经济和人类行为产生重大影响，从而引发对信任ML决策的严重担忧。一些研究表明，传统的度量标准，如模型的可信度和准确性，不足以建立人类对ML模型的信任。这些模型经常在训练过程中学习伪相关性，并在推理过程中基于它们进行预测。在缺乏这种相关性的现实世界中，它们的表现可能会显著恶化。为了避免这种情况，一种常见的做法是测试预测是否合理。随之而来的是一个被称为可信性先知问题的挑战。由于缺乏自动化的可信性先知，评估需要对解释方法披露的决策过程进行人工验证，这既耗时又不可扩展。本文提出了第一种自动生成文本分类器可信性预言的方法TOKI，它通过解释方法和词嵌入的方法自动检查预测贡献词是否与预测类相关。为了证明其实用性，我们引入了一种新的针对TOKI确定的可信性问题的对抗性攻击方法。我们将TOKI与单纯基于模型置信度的天真基线进行比较，该基线使用了6,000个预测的人为事实。我们还比较了TOKI引导的对抗性攻击方法和SOTA对抗性攻击方法A2T。结果表明，依靠预测的不确定性不能区分可信和不可信的预测，TOKI的准确率比朴素基线高142%，TOKI引导的对抗性攻击方法比A2T更有效，扰动更少。



## **12. AdvWeb: Controllable Black-box Attacks on VLM-powered Web Agents**

AdvWeb：对TLR驱动的Web代理的可控黑匣子攻击 cs.CR

15 pages

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.17401v2) [paper-pdf](http://arxiv.org/pdf/2410.17401v2)

**Authors**: Chejian Xu, Mintong Kang, Jiawei Zhang, Zeyi Liao, Lingbo Mo, Mengqi Yuan, Huan Sun, Bo Li

**Abstract**: Vision Language Models (VLMs) have revolutionized the creation of generalist web agents, empowering them to autonomously complete diverse tasks on real-world websites, thereby boosting human efficiency and productivity. However, despite their remarkable capabilities, the safety and security of these agents against malicious attacks remain critically underexplored, raising significant concerns about their safe deployment. To uncover and exploit such vulnerabilities in web agents, we provide AdvWeb, a novel black-box attack framework designed against web agents. AdvWeb trains an adversarial prompter model that generates and injects adversarial prompts into web pages, misleading web agents into executing targeted adversarial actions such as inappropriate stock purchases or incorrect bank transactions, actions that could lead to severe real-world consequences. With only black-box access to the web agent, we train and optimize the adversarial prompter model using DPO, leveraging both successful and failed attack strings against the target agent. Unlike prior approaches, our adversarial string injection maintains stealth and control: (1) the appearance of the website remains unchanged before and after the attack, making it nearly impossible for users to detect tampering, and (2) attackers can modify specific substrings within the generated adversarial string to seamlessly change the attack objective (e.g., purchasing stocks from a different company), enhancing attack flexibility and efficiency. We conduct extensive evaluations, demonstrating that AdvWeb achieves high success rates in attacking SOTA GPT-4V-based VLM agent across various web tasks. Our findings expose critical vulnerabilities in current LLM/VLM-based agents, emphasizing the urgent need for developing more reliable web agents and effective defenses. Our code and data are available at https://ai-secure.github.io/AdvWeb/ .

摘要: 视觉语言模型(VLM)彻底改变了多面手Web代理的创建，使其能够在现实世界的网站上自主完成各种任务，从而提高了人类的效率和生产力。然而，尽管这些代理具有非凡的能力，但其抵御恶意攻击的安全性和安全性仍然严重不足，这引发了人们对其安全部署的严重担忧。为了发现和利用Web代理中的此类漏洞，我们提供了AdvWeb，这是一个针对Web代理设计的新型黑盒攻击框架。AdvWeb训练一种对抗性提示器模型，该模型生成对抗性提示并将其注入网页，误导网络代理执行有针对性的对抗性行动，如不适当的股票购买或不正确的银行交易，这些行动可能会导致严重的现实世界后果。在只有黑盒访问Web代理的情况下，我们使用DPO训练和优化对抗性提示器模型，利用针对目标代理的成功和失败的攻击字符串。与以前的方法不同，我们的敌意字符串注入保持了隐蔽性和可控性：(1)攻击前后网站的外观保持不变，使得用户几乎不可能检测到篡改；(2)攻击者可以修改生成的敌意字符串中的特定子字符串，以无缝更改攻击目标(例如，从不同公司购买股票)，从而增强攻击的灵活性和效率。我们进行了广泛的评估，表明AdvWeb在各种Web任务中攻击基于Sota GPT-4V的VLM代理取得了很高的成功率。我们的发现暴露了当前基于LLM/VLM的代理的严重漏洞，强调了开发更可靠的网络代理和有效防御的迫切需要。我们的代码和数据可在https://ai-secure.github.io/AdvWeb/上获得。



## **13. LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate**

LookHere：具有定向注意力的视觉变形者概括和推断 cs.CV

NeurIPS 2024 Camera Ready

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2405.13985v2) [paper-pdf](http://arxiv.org/pdf/2405.13985v2)

**Authors**: Anthony Fuller, Daniel G. Kyrollos, Yousef Yassin, James R. Green

**Abstract**: High-resolution images offer more information about scenes that can improve model accuracy. However, the dominant model architecture in computer vision, the vision transformer (ViT), cannot effectively leverage larger images without finetuning -- ViTs poorly extrapolate to more patches at test time, although transformers offer sequence length flexibility. We attribute this shortcoming to the current patch position encoding methods, which create a distribution shift when extrapolating.   We propose a drop-in replacement for the position encoding of plain ViTs that restricts attention heads to fixed fields of view, pointed in different directions, using 2D attention masks. Our novel method, called LookHere, provides translation-equivariance, ensures attention head diversity, and limits the distribution shift that attention heads face when extrapolating. We demonstrate that LookHere improves performance on classification (avg. 1.6%), against adversarial attack (avg. 5.4%), and decreases calibration error (avg. 1.5%) -- on ImageNet without extrapolation. With extrapolation, LookHere outperforms the current SoTA position encoding method, 2D-RoPE, by 21.7% on ImageNet when trained at $224^2$ px and tested at $1024^2$ px. Additionally, we release a high-resolution test set to improve the evaluation of high-resolution image classifiers, called ImageNet-HR.

摘要: 高分辨率图像提供了有关场景的更多信息，可以提高模型精度。然而，计算机视觉中占主导地位的模型体系结构视觉转换器(VIT)在没有精细调整的情况下无法有效地利用更大的图像-VIT在测试时很难外推到更多的补丁，尽管转换器提供了序列长度的灵活性。我们将这一缺陷归因于目前的补丁位置编码方法，这些方法在外推时会产生分布偏移。我们提出了一种替代普通VITS的位置编码的方法，它使用2D注意掩码将注意力头部限制在指向不同方向的固定视野中。我们的新方法LookHere提供了平移等差性，确保了注意力头部的多样性，并限制了注意力头部在外推时面临的分布变化。我们证明LookHere提高了分类性能(平均1.6%)，抗对手攻击(Avg.5.4%)，降低了校准误差(平均1.5%)--在ImageNet上，没有外推。通过外推，LookHere在ImageNet上以$224^2$px进行训练并以$1024^2$px进行测试时，在ImageNet上的性能比当前的SOTA位置编码方法2D-ROPE高21.7%。此外，我们还发布了一个高分辨率测试集来改进高分辨率图像分类器的评估，称为ImageNet-HR。



## **14. Power side-channel leakage localization through adversarial training of deep neural networks**

通过深度神经网络的对抗训练进行电源侧通道泄漏定位 cs.LG

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22425v1) [paper-pdf](http://arxiv.org/pdf/2410.22425v1)

**Authors**: Jimmy Gammell, Anand Raghunathan, Kaushik Roy

**Abstract**: Supervised deep learning has emerged as an effective tool for carrying out power side-channel attacks on cryptographic implementations. While increasingly-powerful deep learning-based attacks are regularly published, comparatively-little work has gone into using deep learning to defend against these attacks. In this work we propose a technique for identifying which timesteps in a power trace are responsible for leaking a cryptographic key, through an adversarial game between a deep learning-based side-channel attacker which seeks to classify a sensitive variable from the power traces recorded during encryption, and a trainable noise generator which seeks to thwart this attack by introducing a minimal amount of noise into the power traces. We demonstrate on synthetic datasets that our method can outperform existing techniques in the presence of common countermeasures such as Boolean masking and trace desynchronization. Results on real datasets are weak because the technique is highly sensitive to hyperparameters and early-stop point, and we lack a holdout dataset with ground truth knowledge of leaking points for model selection. Nonetheless, we believe our work represents an important first step towards deep side-channel leakage localization without relying on strong assumptions about the implementation or the nature of its leakage. An open-source PyTorch implementation of our experiments is provided.

摘要: 有监督的深度学习已经成为对密码实现进行功率侧通道攻击的有效工具。虽然越来越强大的基于深度学习的攻击定期发布，但相对较少的工作是使用深度学习来防御这些攻击。在这项工作中，我们提出了一种技术，通过基于深度学习的侧通道攻击者和可训练噪声生成器之间的对抗性博弈来识别功率跟踪中的哪些时间步骤负责泄漏密钥，侧通道攻击者试图从加密过程中记录的功率跟踪中将敏感变量分类，而可训练噪声生成器试图通过在功率跟踪中引入最少量的噪声来阻止这种攻击。我们在合成数据集上演示了我们的方法在存在布尔掩蔽和跟踪去同步等常见对策的情况下可以优于现有的技术。在真实数据集上的结果很弱，因为该技术对超参数和提前停止点高度敏感，并且我们缺乏一个具有泄漏点基本事实知识的数据集来选择模型。尽管如此，我们相信我们的工作代表着迈向深侧沟道泄漏定位的重要的第一步，而不依赖于对其实施或其泄漏的性质的强烈假设。给出了我们实验的一个开源的PyTorch实现。



## **15. SVIP: Towards Verifiable Inference of Open-source Large Language Models**

SVIP：迈向开源大型语言模型的可验证推理 cs.LG

20 pages

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22307v1) [paper-pdf](http://arxiv.org/pdf/2410.22307v1)

**Authors**: Yifan Sun, Yuhang Li, Yue Zhang, Yuchen Jin, Huan Zhang

**Abstract**: Open-source Large Language Models (LLMs) have recently demonstrated remarkable capabilities in natural language understanding and generation, leading to widespread adoption across various domains. However, their increasing model sizes render local deployment impractical for individual users, pushing many to rely on computing service providers for inference through a blackbox API. This reliance introduces a new risk: a computing provider may stealthily substitute the requested LLM with a smaller, less capable model without consent from users, thereby delivering inferior outputs while benefiting from cost savings. In this paper, we formalize the problem of verifiable inference for LLMs. Existing verifiable computing solutions based on cryptographic or game-theoretic techniques are either computationally uneconomical or rest on strong assumptions. We introduce SVIP, a secret-based verifiable LLM inference protocol that leverages intermediate outputs from LLM as unique model identifiers. By training a proxy task on these outputs and requiring the computing provider to return both the generated text and the processed intermediate outputs, users can reliably verify whether the computing provider is acting honestly. In addition, the integration of a secret mechanism further enhances the security of our protocol. We thoroughly analyze our protocol under multiple strong and adaptive adversarial scenarios. Our extensive experiments demonstrate that SVIP is accurate, generalizable, computationally efficient, and resistant to various attacks. Notably, SVIP achieves false negative rates below 5% and false positive rates below 3%, while requiring less than 0.01 seconds per query for verification.

摘要: 开源的大型语言模型(LLM)最近在自然语言理解和生成方面表现出了非凡的能力，导致了在各个领域的广泛采用。然而，它们不断增长的模型规模使得本地部署对个人用户来说是不现实的，促使许多人依赖计算服务提供商通过黑盒API进行推理。这种依赖带来了新的风险：计算提供商可能会在未经用户同意的情况下，悄悄地用较小、功能较差的模型替换所请求的LLM，从而在提供劣质产出的同时受益于成本节约。本文对LLMS的可验证推理问题进行了形式化描述。现有的基于密码学或博弈论技术的可验证计算解决方案要么在计算上不经济，要么依赖于强有力的假设。我们引入了SVIP，这是一个基于秘密的可验证LLM推理协议，它利用LLM的中间输出作为唯一的模型标识符。通过对这些输出训练代理任务并要求计算提供商返回生成的文本和处理的中间输出，用户可以可靠地验证计算提供商是否诚实行事。此外，秘密机制的集成进一步增强了协议的安全性。我们深入分析了我们的协议在多种强和自适应对抗场景下的性能。大量实验表明，SVIP算法具有较高的准确性、通用性、计算效率和抵抗各种攻击的能力。值得注意的是，SVIP实现了5%以下的假阴性率和3%以下的假阳性率，而每次查询验证所需的时间不到0.01秒。



## **16. Embedding-based classifiers can detect prompt injection attacks**

基于嵌入的分类器可以检测提示注入攻击 cs.CR

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22284v1) [paper-pdf](http://arxiv.org/pdf/2410.22284v1)

**Authors**: Md. Ahsan Ayub, Subhabrata Majumdar

**Abstract**: Large Language Models (LLMs) are seeing significant adoption in every type of organization due to their exceptional generative capabilities. However, LLMs are found to be vulnerable to various adversarial attacks, particularly prompt injection attacks, which trick them into producing harmful or inappropriate content. Adversaries execute such attacks by crafting malicious prompts to deceive the LLMs. In this paper, we propose a novel approach based on embedding-based Machine Learning (ML) classifiers to protect LLM-based applications against this severe threat. We leverage three commonly used embedding models to generate embeddings of malicious and benign prompts and utilize ML classifiers to predict whether an input prompt is malicious. Out of several traditional ML methods, we achieve the best performance with classifiers built using Random Forest and XGBoost. Our classifiers outperform state-of-the-art prompt injection classifiers available in open-source implementations, which use encoder-only neural networks.

摘要: 大型语言模型(LLM)由于其非凡的生成能力，在每种类型的组织中都得到了大量采用。然而，LLM被发现容易受到各种对抗性攻击，特别是即时注入攻击，这些攻击会诱使它们产生有害或不适当的内容。攻击者通过精心编制恶意提示来欺骗LLM，从而执行此类攻击。在本文中，我们提出了一种基于嵌入的机器学习(ML)分类器的新方法来保护基于LLM的应用程序免受这种严重威胁。我们利用三种常用的嵌入模型来生成恶意提示和良性提示的嵌入，并利用ML分类器来预测输入提示是否为恶意提示。在几种传统的最大似然分类方法中，我们使用随机森林和XGBoost构建的分类器取得了最好的性能。我们的分类器比开源实现中可用的最先进的提示注入分类器性能更好，后者使用仅限编码器的神经网络。



## **17. AmpleGCG-Plus: A Strong Generative Model of Adversarial Suffixes to Jailbreak LLMs with Higher Success Rates in Fewer Attempts**

AmpleGCG-Plus：越狱LLC的对抗性后缀的强生成模型，以更少的尝试获得更高的成功率 cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22143v1) [paper-pdf](http://arxiv.org/pdf/2410.22143v1)

**Authors**: Vishal Kumar, Zeyi Liao, Jaylen Jones, Huan Sun

**Abstract**: Although large language models (LLMs) are typically aligned, they remain vulnerable to jailbreaking through either carefully crafted prompts in natural language or, interestingly, gibberish adversarial suffixes. However, gibberish tokens have received relatively less attention despite their success in attacking aligned LLMs. Recent work, AmpleGCG~\citep{liao2024amplegcg}, demonstrates that a generative model can quickly produce numerous customizable gibberish adversarial suffixes for any harmful query, exposing a range of alignment gaps in out-of-distribution (OOD) language spaces. To bring more attention to this area, we introduce AmpleGCG-Plus, an enhanced version that achieves better performance in fewer attempts. Through a series of exploratory experiments, we identify several training strategies to improve the learning of gibberish suffixes. Our results, verified under a strict evaluation setting, show that it outperforms AmpleGCG on both open-weight and closed-source models, achieving increases in attack success rate (ASR) of up to 17\% in the white-box setting against Llama-2-7B-chat, and more than tripling ASR in the black-box setting against GPT-4. Notably, AmpleGCG-Plus jailbreaks the newer GPT-4o series of models at similar rates to GPT-4, and, uncovers vulnerabilities against the recently proposed circuit breakers defense. We publicly release AmpleGCG-Plus along with our collected training datasets.

摘要: 尽管大型语言模型(LLM)通常是一致的，但它们仍然很容易通过精心设计的自然语言提示或有趣的胡言乱语对抗性后缀越狱。然而，令人费解的令牌尽管成功地攻击了对齐的LLM，但受到的关注相对较少。最近的工作，AmpleGCG~\Citep{Lio2024Amplegcg}，证明了生成模型可以为任何有害的查询快速生成大量可定制的胡言乱语对抗性后缀，从而暴露出分布外(OOD)语言空间中的一系列对齐差距。为了引起人们对这一领域的更多关注，我们推出了AmpleGCG-Plus，这是一个增强版本，在较少的尝试中获得了更好的性能。通过一系列的探索性实验，我们确定了几种训练策略来提高乱码后缀的学习效果。在严格的评估设置下验证的结果表明，它在开源和闭源模型上都优于AmpleGCG，在白盒环境下相对于Llama-2-7B-Chat的攻击成功率(ASR)提高了17%，在黑盒环境下相对于GPT-4的攻击成功率(ASR)提高了两倍以上。值得注意的是，AmpleGCG-Plus以类似于GPT-4的速度监禁了较新的GPT-4o系列型号，并揭示了针对最近提出的断路器防御的漏洞。我们公开发布AmpleGCG-Plus以及我们收集的训练数据集。



## **18. Iterative Window Mean Filter: Thwarting Diffusion-based Adversarial Purification**

迭代窗口均值过滤器：阻止基于扩散的对抗净化 cs.CR

Accepted in IEEE Transactions on Dependable and Secure Computing

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2408.10673v3) [paper-pdf](http://arxiv.org/pdf/2408.10673v3)

**Authors**: Hanrui Wang, Ruoxi Sun, Cunjian Chen, Minhui Xue, Lay-Ki Soon, Shuo Wang, Zhe Jin

**Abstract**: Face authentication systems have brought significant convenience and advanced developments, yet they have become unreliable due to their sensitivity to inconspicuous perturbations, such as adversarial attacks. Existing defenses often exhibit weaknesses when facing various attack algorithms and adaptive attacks or compromise accuracy for enhanced security. To address these challenges, we have developed a novel and highly efficient non-deep-learning-based image filter called the Iterative Window Mean Filter (IWMF) and proposed a new framework for adversarial purification, named IWMF-Diff, which integrates IWMF and denoising diffusion models. These methods can function as pre-processing modules to eliminate adversarial perturbations without necessitating further modifications or retraining of the target system. We demonstrate that our proposed methodologies fulfill four critical requirements: preserved accuracy, improved security, generalizability to various threats in different settings, and better resistance to adaptive attacks. This performance surpasses that of the state-of-the-art adversarial purification method, DiffPure.

摘要: 人脸认证系统带来了极大的便利和先进的发展，但由于它们对诸如敌意攻击等不起眼的扰动非常敏感，因此变得不可靠。现有的防御在面对各种攻击算法和自适应攻击时往往表现出弱点，或者为了增强安全性而损害准确性。为了应对这些挑战，我们开发了一种新颖高效的基于非深度学习的图像过滤器，称为迭代窗口均值过滤器(IWMF)，并提出了一种结合IWMF和去噪扩散模型的新的对抗性净化框架IWMF-DIFF。这些方法可以作为前处理模块来消除对抗性干扰，而不需要对目标系统进行进一步的修改或重新培训。我们证明了我们提出的方法满足了四个关键要求：保持准确性，提高安全性，对不同环境下的各种威胁具有通用性，以及更好地抵抗自适应攻击。这一性能超过了最先进的对抗性净化方法DiffPure。



## **19. Forging the Forger: An Attempt to Improve Authorship Verification via Data Augmentation**

伪造伪造者：通过数据增强改进作者身份验证的尝试 cs.LG

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2403.11265v2) [paper-pdf](http://arxiv.org/pdf/2403.11265v2)

**Authors**: Silvia Corbara, Alejandro Moreo

**Abstract**: Authorship Verification (AV) is a text classification task concerned with inferring whether a candidate text has been written by one specific author or by someone else. It has been shown that many AV systems are vulnerable to adversarial attacks, where a malicious author actively tries to fool the classifier by either concealing their writing style, or by imitating the style of another author. In this paper, we investigate the potential benefits of augmenting the classifier training set with (negative) synthetic examples. These synthetic examples are generated to imitate the style of the author of interest. We analyze the improvements in classifier prediction that this augmentation brings to bear in the task of AV in an adversarial setting. In particular, we experiment with three different generator architectures (one based on Recurrent Neural Networks, another based on small-scale transformers, and another based on the popular GPT model) and with two training strategies (one inspired by standard Language Models, and another inspired by Wasserstein Generative Adversarial Networks). We evaluate our hypothesis on five datasets (three of which have been specifically collected to represent an adversarial setting) and using two learning algorithms for the AV classifier (Support Vector Machines and Convolutional Neural Networks). This experimentation has yielded negative results, revealing that, although our methodology proves effective in many adversarial settings, its benefits are too sporadic for a pragmatical application.

摘要: 作者身份验证是一项文本分类任务，涉及推断候选文本是由某个特定作者还是其他人撰写的。已经证明，许多反病毒系统容易受到敌意攻击，恶意作者通过隐藏他们的写作风格或模仿另一位作者的风格来主动试图愚弄分类器。在本文中，我们研究了用(负的)合成例子来扩大分类器训练集的潜在好处。这些合成的例子是为了模仿感兴趣的作者的风格而产生的。我们分析了这种增强在对抗性环境下对AV任务带来的分类器预测方面的改进。特别是，我们实验了三种不同的生成器体系结构(一种基于递归神经网络，另一种基于小型变压器，另一种基于流行的GPT模型)和两种训练策略(一种受到标准语言模型的启发，另一种受到Wasserstein生成性对手网络的启发)。我们在五个数据集(其中三个已经被专门收集来表示对抗性环境)上评估了我们的假设，并使用了两种用于AV分类器的学习算法(支持向量机和卷积神经网络)。这种实验产生了负面的结果，表明尽管我们的方法在许多对抗性环境中被证明是有效的，但它的好处对于实用应用来说太零星了。



## **20. On the Robustness of Adversarial Training Against Uncertainty Attacks**

论对抗训练对不确定性攻击的鲁棒性 cs.LG

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21952v1) [paper-pdf](http://arxiv.org/pdf/2410.21952v1)

**Authors**: Emanuele Ledda, Giovanni Scodeller, Daniele Angioni, Giorgio Piras, Antonio Emanuele Cinà, Giorgio Fumera, Battista Biggio, Fabio Roli

**Abstract**: In learning problems, the noise inherent to the task at hand hinders the possibility to infer without a certain degree of uncertainty. Quantifying this uncertainty, regardless of its wide use, assumes high relevance for security-sensitive applications. Within these scenarios, it becomes fundamental to guarantee good (i.e., trustworthy) uncertainty measures, which downstream modules can securely employ to drive the final decision-making process. However, an attacker may be interested in forcing the system to produce either (i) highly uncertain outputs jeopardizing the system's availability or (ii) low uncertainty estimates, making the system accept uncertain samples that would instead require a careful inspection (e.g., human intervention). Therefore, it becomes fundamental to understand how to obtain robust uncertainty estimates against these kinds of attacks. In this work, we reveal both empirically and theoretically that defending against adversarial examples, i.e., carefully perturbed samples that cause misclassification, additionally guarantees a more secure, trustworthy uncertainty estimate under common attack scenarios without the need for an ad-hoc defense strategy. To support our claims, we evaluate multiple adversarial-robust models from the publicly available benchmark RobustBench on the CIFAR-10 and ImageNet datasets.

摘要: 在学习问题中，手头任务固有的噪音阻碍了在没有一定程度的不确定性的情况下进行推断的可能性。量化这种不确定性，不管它是否被广泛使用，都假定它与安全敏感的应用程序高度相关。在这些场景中，保证良好的(即值得信赖的)不确定性度量变得至关重要，下游模块可以安全地使用这些度量来驱动最终的决策过程。然而，攻击者可能有兴趣强迫系统产生(I)危及系统可用性的高度不确定的输出，或(Ii)低不确定性估计，使系统接受不确定的样本，而不是需要仔细检查(例如，人工干预)。因此，了解如何针对这类攻击获得稳健的不确定性估计变得至关重要。在这项工作中，我们从经验和理论上揭示了对敌意示例的防御，即仔细扰动导致错误分类的样本，额外地保证了在常见攻击场景下更安全、更可信的不确定性估计，而不需要特别的防御策略。为了支持我们的主张，我们在CIFAR-10和ImageNet数据集上评估了来自公开可用的基准RobustBch的多个对抗性稳健模型。



## **21. Enhancing Adversarial Attacks through Chain of Thought**

通过思维链增强对抗性攻击 cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21791v1) [paper-pdf](http://arxiv.org/pdf/2410.21791v1)

**Authors**: Jingbo Su

**Abstract**: Large language models (LLMs) have demonstrated impressive performance across various domains but remain susceptible to safety concerns. Prior research indicates that gradient-based adversarial attacks are particularly effective against aligned LLMs and the chain of thought (CoT) prompting can elicit desired answers through step-by-step reasoning. This paper proposes enhancing the robustness of adversarial attacks on aligned LLMs by integrating CoT prompts with the greedy coordinate gradient (GCG) technique. Using CoT triggers instead of affirmative targets stimulates the reasoning abilities of backend LLMs, thereby improving the transferability and universality of adversarial attacks. We conducted an ablation study comparing our CoT-GCG approach with Amazon Web Services auto-cot. Results revealed our approach outperformed both the baseline GCG attack and CoT prompting. Additionally, we used Llama Guard to evaluate potentially harmful interactions, providing a more objective risk assessment of entire conversations compared to matching outputs to rejection phrases. The code of this paper is available at https://github.com/sujingbo0217/CS222W24-LLM-Attack.

摘要: 大型语言模型(LLM)在各个领域都表现出了令人印象深刻的表现，但仍然容易受到安全问题的影响。以往的研究表明，基于梯度的对抗性攻击对于对齐的LLM特别有效，而思维链(COT)提示可以通过循序渐进的推理获得期望的答案。提出了一种将CoT提示与贪婪坐标梯度(GCG)技术相结合的方法，以增强对齐LLMS的敌意攻击的稳健性。使用CoT触发器代替肯定目标，刺激了后端LLMS的推理能力，从而提高了对抗性攻击的可转移性和通用性。我们进行了一项烧蚀研究，将我们的COT-GCG方法与Amazon Web Services自动COT进行了比较。结果显示，我们的方法比基线GCG攻击和COT提示都要好。此外，我们使用Llama Guard来评估潜在的有害交互，与将输出与拒绝短语匹配相比，提供了对整个对话的更客观的风险评估。本文的代码可在https://github.com/sujingbo0217/CS222W24-LLM-Attack.上找到



## **22. Transferable Adversarial Attacks on SAM and Its Downstream Models**

对Sam及其下游模型的可转移对抗攻击 cs.LG

This work is accepted by Neurips2024

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.20197v2) [paper-pdf](http://arxiv.org/pdf/2410.20197v2)

**Authors**: Song Xia, Wenhan Yang, Yi Yu, Xun Lin, Henghui Ding, Lingyu Duan, Xudong Jiang

**Abstract**: The utilization of large foundational models has a dilemma: while fine-tuning downstream tasks from them holds promise for making use of the well-generalized knowledge in practical applications, their open accessibility also poses threats of adverse usage. This paper, for the first time, explores the feasibility of adversarial attacking various downstream models fine-tuned from the segment anything model (SAM), by solely utilizing the information from the open-sourced SAM. In contrast to prevailing transfer-based adversarial attacks, we demonstrate the existence of adversarial dangers even without accessing the downstream task and dataset to train a similar surrogate model. To enhance the effectiveness of the adversarial attack towards models fine-tuned on unknown datasets, we propose a universal meta-initialization (UMI) algorithm to extract the intrinsic vulnerability inherent in the foundation model, which is then utilized as the prior knowledge to guide the generation of adversarial perturbations. Moreover, by formulating the gradient difference in the attacking process between the open-sourced SAM and its fine-tuned downstream models, we theoretically demonstrate that a deviation occurs in the adversarial update direction by directly maximizing the distance of encoded feature embeddings in the open-sourced SAM. Consequently, we propose a gradient robust loss that simulates the associated uncertainty with gradient-based noise augmentation to enhance the robustness of generated adversarial examples (AEs) towards this deviation, thus improving the transferability. Extensive experiments demonstrate the effectiveness of the proposed universal meta-initialized and gradient robust adversarial attack (UMI-GRAT) toward SAMs and their downstream models. Code is available at https://github.com/xiasong0501/GRAT.

摘要: 大型基础模型的利用有一个两难的境地：虽然从它们微调下游任务有望在实际应用中利用良好的通用知识，但它们的开放可访问性也构成了不利使用的威胁。本文首次探索了仅利用开源SAM的信息，对从段任何模型(SAM)微调而来的各种下游模型进行对抗性攻击的可行性。与目前流行的基于转移的对抗性攻击相比，我们证明了即使在没有访问下游任务和数据集来训练类似的代理模型的情况下，也存在对抗性危险。为了提高对未知数据集精调模型的敌意攻击的有效性，我们提出了一种通用的元初始化(UMI)算法来提取基础模型中固有的脆弱性，并将其作为先验知识来指导敌意扰动的生成。此外，通过描述开源SAM及其微调下游模型在攻击过程中的梯度差，我们从理论上证明了通过直接最大化开源SAM中编码特征嵌入的距离，在对抗性更新方向上发生了偏差。因此，我们提出了一种梯度稳健损失，它通过基于梯度的噪声增强来模拟关联的不确定性，以增强生成的对抗性实例(AES)对这种偏差的稳健性，从而提高了可转移性。大量实验证明了该算法对地对空导弹及其下游模型的有效攻击。代码可在https://github.com/xiasong0501/GRAT.上找到



## **23. CFSafety: Comprehensive Fine-grained Safety Assessment for LLMs**

CFSafety：LLM的全面细粒度安全评估 cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21695v1) [paper-pdf](http://arxiv.org/pdf/2410.21695v1)

**Authors**: Zhihao Liu, Chenhui Hu

**Abstract**: As large language models (LLMs) rapidly evolve, they bring significant conveniences to our work and daily lives, but also introduce considerable safety risks. These models can generate texts with social biases or unethical content, and under specific adversarial instructions, may even incite illegal activities. Therefore, rigorous safety assessments of LLMs are crucial. In this work, we introduce a safety assessment benchmark, CFSafety, which integrates 5 classic safety scenarios and 5 types of instruction attacks, totaling 10 categories of safety questions, to form a test set with 25k prompts. This test set was used to evaluate the natural language generation (NLG) capabilities of LLMs, employing a combination of simple moral judgment and a 1-5 safety rating scale for scoring. Using this benchmark, we tested eight popular LLMs, including the GPT series. The results indicate that while GPT-4 demonstrated superior safety performance, the safety effectiveness of LLMs, including this model, still requires improvement. The data and code associated with this study are available on GitHub.

摘要: 随着大型语言模型的快速发展，它们在给我们的工作和日常生活带来极大便利的同时，也带来了相当大的安全隐患。这些模式可能会产生带有社会偏见或不道德内容的文本，在特定的敌对指令下，甚至可能煽动非法活动。因此，对LLMS进行严格的安全评估至关重要。在这项工作中，我们引入了一个安全评估基准CFSafe，它集成了5个经典的安全场景和5种类型的指令攻击，共计10类安全问题，形成了一个包含25K提示的测试集。该测试集被用来评估LLMS的自然语言生成(NLG)能力，采用简单的道德判断和1-5安全等级评分相结合的方式进行评分。使用这个基准，我们测试了八个流行的LLM，包括GPT系列。结果表明，尽管GPT-4表现出了优越的安全性能，但包括该模型在内的LLMS的安全有效性仍需改进。与这项研究相关的数据和代码可在GitHub上获得。



## **24. AdvI2I: Adversarial Image Attack on Image-to-Image Diffusion models**

AdvI 2I：对图像到图像扩散模型的对抗图像攻击 cs.CV

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21471v1) [paper-pdf](http://arxiv.org/pdf/2410.21471v1)

**Authors**: Yaopei Zeng, Yuanpu Cao, Bochuan Cao, Yurui Chang, Jinghui Chen, Lu Lin

**Abstract**: Recent advances in diffusion models have significantly enhanced the quality of image synthesis, yet they have also introduced serious safety concerns, particularly the generation of Not Safe for Work (NSFW) content. Previous research has demonstrated that adversarial prompts can be used to generate NSFW content. However, such adversarial text prompts are often easily detectable by text-based filters, limiting their efficacy. In this paper, we expose a previously overlooked vulnerability: adversarial image attacks targeting Image-to-Image (I2I) diffusion models. We propose AdvI2I, a novel framework that manipulates input images to induce diffusion models to generate NSFW content. By optimizing a generator to craft adversarial images, AdvI2I circumvents existing defense mechanisms, such as Safe Latent Diffusion (SLD), without altering the text prompts. Furthermore, we introduce AdvI2I-Adaptive, an enhanced version that adapts to potential countermeasures and minimizes the resemblance between adversarial images and NSFW concept embeddings, making the attack more resilient against defenses. Through extensive experiments, we demonstrate that both AdvI2I and AdvI2I-Adaptive can effectively bypass current safeguards, highlighting the urgent need for stronger security measures to address the misuse of I2I diffusion models.

摘要: 扩散模型的最新进展显著提高了图像合成的质量，但也引入了严重的安全问题，特别是不安全工作(NSFW)内容的产生。以前的研究已经证明，对抗性提示可以用来生成NSFW内容。然而，这种对抗性的文本提示通常很容易被基于文本的过滤器检测到，从而限制了它们的有效性。在本文中，我们暴露了一个以前被忽视的漏洞：针对图像到图像(I2I)扩散模型的对抗性图像攻击。我们提出了AdvI2I，这是一个新的框架，它通过操作输入图像来诱导扩散模型来生成NSFW内容。通过优化生成器来制作敌意图像，AdvI2I在不改变文本提示的情况下绕过了现有的防御机制，如安全潜在扩散(SLD)。此外，我们引入了AdvI2I-自适应，这是一个增强版本，它适应潜在的对策，并将敌对图像和NSFW概念嵌入之间的相似性降至最低，使攻击对防御更具弹性。通过广泛的实验，我们证明了AdvI2I和AdvI2I-自适应都可以有效地绕过当前的保障措施，这突显了迫切需要更强大的安全措施来解决I2I扩散模型的滥用。



## **25. TACO: Adversarial Camouflage Optimization on Trucks to Fool Object Detectors**

TACO：卡车对抗性伪装优化以愚弄物体检测器 cs.CV

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21443v1) [paper-pdf](http://arxiv.org/pdf/2410.21443v1)

**Authors**: Adonisz Dimitriu, Tamás Michaletzky, Viktor Remeli

**Abstract**: Adversarial attacks threaten the reliability of machine learning models in critical applications like autonomous vehicles and defense systems. As object detectors become more robust with models like YOLOv8, developing effective adversarial methodologies is increasingly challenging. We present Truck Adversarial Camouflage Optimization (TACO), a novel framework that generates adversarial camouflage patterns on 3D vehicle models to deceive state-of-the-art object detectors. Adopting Unreal Engine 5, TACO integrates differentiable rendering with a Photorealistic Rendering Network to optimize adversarial textures targeted at YOLOv8. To ensure the generated textures are both effective in deceiving detectors and visually plausible, we introduce the Convolutional Smooth Loss function, a generalized smooth loss function. Experimental evaluations demonstrate that TACO significantly degrades YOLOv8's detection performance, achieving an AP@0.5 of 0.0099 on unseen test data. Furthermore, these adversarial patterns exhibit strong transferability to other object detection models such as Faster R-CNN and earlier YOLO versions.

摘要: 对抗性攻击威胁着机器学习模型在自动驾驶车辆和防御系统等关键应用中的可靠性。随着像YOLOv8这样的模型使目标探测器变得更加健壮，开发有效的对抗性方法越来越具有挑战性。提出了卡车对抗性伪装优化(TACO)，这是一种在3D车辆模型上生成对抗性伪装图案以欺骗最先进的目标检测器的新框架。采用虚幻引擎5，Taco集成了可区分渲染和照片级真实感渲染网络，以优化针对YOLOv8的对抗性纹理。为了确保生成的纹理既能有效地欺骗检测器，又能在视觉上可信，我们引入了卷积平滑损失函数，这是一种广义的平滑损失函数。实验评估表明，TACO显著降低了YOLOv8的S检测性能，在未知测试数据上的AP@0.5%为0.0099。此外，这些对抗性模式表现出很强的可移植到其他目标检测模型，如更快的R-CNN和更早的YOLO版本。



## **26. Securing Multi-turn Conversational Language Models From Distributed Backdoor Triggers**

保护多轮对话语言模型免受分布式后门触发器的影响 cs.CL

Findings of EMNLP 2024

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2407.04151v2) [paper-pdf](http://arxiv.org/pdf/2407.04151v2)

**Authors**: Terry Tong, Jiashu Xu, Qin Liu, Muhao Chen

**Abstract**: Large language models (LLMs) have acquired the ability to handle longer context lengths and understand nuances in text, expanding their dialogue capabilities beyond a single utterance. A popular user-facing application of LLMs is the multi-turn chat setting. Though longer chat memory and better understanding may seemingly benefit users, our paper exposes a vulnerability that leverages the multi-turn feature and strong learning ability of LLMs to harm the end-user: the backdoor. We demonstrate that LLMs can capture the combinational backdoor representation. Only upon presentation of triggers together does the backdoor activate. We also verify empirically that this representation is invariant to the position of the trigger utterance. Subsequently, inserting a single extra token into two utterances of 5%of the data can cause over 99% Attack Success Rate (ASR). Our results with 3 triggers demonstrate that this framework is generalizable, compatible with any trigger in an adversary's toolbox in a plug-and-play manner. Defending the backdoor can be challenging in the chat setting because of the large input and output space. Our analysis indicates that the distributed backdoor exacerbates the current challenges by polynomially increasing the dimension of the attacked input space. Canonical textual defenses like ONION and BKI leverage auxiliary model forward passes over individual tokens, scaling exponentially with the input sequence length and struggling to maintain computational feasibility. To this end, we propose a decoding time defense - decayed contrastive decoding - that scales linearly with assistant response sequence length and reduces the backdoor to as low as 0.35%.

摘要: 大型语言模型(LLM)已经具备了处理更长的上下文长度和理解文本中的细微差别的能力，将它们的对话能力扩展到了单一话语之外。LLMS的一个流行的面向用户的应用是多轮聊天设置。虽然更长的聊天记忆和更好的理解似乎对用户有利，但我们的论文暴露了一个漏洞，该漏洞利用LLMS的多回合功能和强大的学习能力来伤害最终用户：后门。我们证明了LLMS能够捕获组合后门表示。只有在一起显示触发器时，后门才会激活。我们还通过实验验证了该表示与触发话语的位置不变。随后，在两个5%的数据发声中插入一个额外令牌可以导致超过99%的攻击成功率(ASR)。我们对3个触发器的测试结果表明，该框架具有通用性，以即插即用的方式兼容对手工具箱中的任何触发器。在聊天环境中，由于输入和输出空间很大，保护后门可能是一件具有挑战性的事情。我们的分析表明，分布式后门通过以多项式增加被攻击输入空间的维度来加剧当前的挑战。像Onion和BKI这样的规范文本防御机制利用辅助模型向前传递单个令牌，随着输入序列长度呈指数级扩展，并努力保持计算的可行性。为此，我们提出了一种译码时间防御机制--衰落对比译码，它与辅助响应序列长度成线性关系，并将后门降低到0.35%。



## **27. Resilience in Knowledge Graph Embeddings**

知识图谱嵌入中的弹性 cs.LG

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21163v1) [paper-pdf](http://arxiv.org/pdf/2410.21163v1)

**Authors**: Arnab Sharma, N'Dah Jean Kouagou, Axel-Cyrille Ngonga Ngomo

**Abstract**: In recent years, knowledge graphs have gained interest and witnessed widespread applications in various domains, such as information retrieval, question-answering, recommendation systems, amongst others. Large-scale knowledge graphs to this end have demonstrated their utility in effectively representing structured knowledge. To further facilitate the application of machine learning techniques, knowledge graph embedding (KGE) models have been developed. Such models can transform entities and relationships within knowledge graphs into vectors. However, these embedding models often face challenges related to noise, missing information, distribution shift, adversarial attacks, etc. This can lead to sub-optimal embeddings and incorrect inferences, thereby negatively impacting downstream applications. While the existing literature has focused so far on adversarial attacks on KGE models, the challenges related to the other critical aspects remain unexplored. In this paper, we, first of all, give a unified definition of resilience, encompassing several factors such as generalisation, performance consistency, distribution adaption, and robustness. After formalizing these concepts for machine learning in general, we define them in the context of knowledge graphs. To find the gap in the existing works on resilience in the context of knowledge graphs, we perform a systematic survey, taking into account all these aspects mentioned previously. Our survey results show that most of the existing works focus on a specific aspect of resilience, namely robustness. After categorizing such works based on their respective aspects of resilience, we discuss the challenges and future research directions.

摘要: 近年来，知识图在信息检索、问答、推荐系统等领域受到了广泛的关注和广泛的应用。为此，大规模的知识图已经证明了它们在有效表示结构化知识方面的有效性。为了进一步促进机器学习技术的应用，人们开发了知识图嵌入(KGE)模型。这样的模型可以将知识图中的实体和关系转换为向量。然而，这些嵌入模型经常面临与噪声、信息缺失、分布漂移、对抗性攻击等相关的挑战，这可能导致次优嵌入和错误推断，从而对下游应用产生负面影响。虽然现有的文献到目前为止都集中在对KGE模型的对抗性攻击上，但与其他关键方面相关的挑战仍然没有被探索。在本文中，我们首先给出了弹性的统一定义，包括泛化、性能一致性、分布适应性和稳健性等因素。在将这些概念形式化之后，我们将在知识图的上下文中定义它们。为了找出现有关于知识图背景下韧性的研究中的差距，我们进行了系统的调查，考虑了前面提到的所有这些方面。我们的调查结果表明，现有的大部分工作都集中在弹性的一个特定方面，即鲁棒性。在对这类研究进行分类后，我们讨论了这些研究面临的挑战和未来的研究方向。



## **28. Adversarial robustness of VAEs through the lens of local geometry**

从局部几何角度来看VAE的对抗鲁棒性 cs.LG

International Conference on Artificial Intelligence and Statistics  (AISTATS) 2023

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2208.03923v3) [paper-pdf](http://arxiv.org/pdf/2208.03923v3)

**Authors**: Asif Khan, Amos Storkey

**Abstract**: In an unsupervised attack on variational autoencoders (VAEs), an adversary finds a small perturbation in an input sample that significantly changes its latent space encoding, thereby compromising the reconstruction for a fixed decoder. A known reason for such vulnerability is the distortions in the latent space resulting from a mismatch between approximated latent posterior and a prior distribution. Consequently, a slight change in an input sample can move its encoding to a low/zero density region in the latent space resulting in an unconstrained generation. This paper demonstrates that an optimal way for an adversary to attack VAEs is to exploit a directional bias of a stochastic pullback metric tensor induced by the encoder and decoder networks. The pullback metric tensor of an encoder measures the change in infinitesimal latent volume from an input to a latent space. Thus, it can be viewed as a lens to analyse the effect of input perturbations leading to latent space distortions. We propose robustness evaluation scores using the eigenspectrum of a pullback metric tensor. Moreover, we empirically show that the scores correlate with the robustness parameter $\beta$ of the $\beta-$VAE. Since increasing $\beta$ also degrades reconstruction quality, we demonstrate a simple alternative using \textit{mixup} training to fill the empty regions in the latent space, thus improving robustness with improved reconstruction.

摘要: 在对变分自动编码器(VAE)的无监督攻击中，攻击者发现输入样本中的微小扰动显著改变了其潜在空间编码，从而危及固定解码器的重建。造成这种脆弱性的一个已知原因是由于近似的潜在后验分布和先验分布之间的不匹配而导致的潜在空间中的扭曲。因此，输入样本中的微小变化可以将其编码移动到潜在空间中的低/零密度区域，从而产生不受限制的生成。证明了敌手攻击VAE的最佳方法是利用编解码网引起的随机拉回度量张量的方向偏差。编码器的回拉度量张量测量从输入到潜在空间的无穷小潜在体积的变化。因此，可以将其视为分析输入扰动导致潜在空间扭曲的影响的透镜。我们使用拉回度量张量的特征谱来提出稳健性评价分数。此外，我们的经验表明，得分与$\beta-$VAE的稳健性参数$\beta$相关。由于增加$\beta$也会降低重建质量，我们演示了一种简单的替代方法，使用文本{Mixup}训练来填充潜在空间中的空区域，从而通过改进重建来提高鲁棒性。



## **29. Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models**

使用语言模型生成的对抗性示例进行攻击错误信息检测 cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.20940v1) [paper-pdf](http://arxiv.org/pdf/2410.20940v1)

**Authors**: Piotr Przybyła

**Abstract**: We investigate the challenge of generating adversarial examples to test the robustness of text classification algorithms detecting low-credibility content, including propaganda, false claims, rumours and hyperpartisan news. We focus on simulation of content moderation by setting realistic limits on the number of queries an attacker is allowed to attempt. Within our solution (TREPAT), initial rephrasings are generated by large language models with prompts inspired by meaning-preserving NLP tasks, e.g. text simplification and style transfer. Subsequently, these modifications are decomposed into small changes, applied through beam search procedure until the victim classifier changes its decision. The evaluation confirms the superiority of our approach in the constrained scenario, especially in case of long input text (news articles), where exhaustive search is not feasible.

摘要: 我们调查了生成敌对示例的挑战，以测试检测低可信度内容（包括宣传、虚假声明、谣言和超党派新闻）的文本分类算法的稳健性。我们通过对允许攻击者尝试的查询数量设置现实的限制来重点模拟内容审核。在我们的解决方案（TREPAT）中，初始改写由大型语言模型生成，其提示受到保留意义的NLP任务（例如文本简化和风格转移）的启发。随后，这些修改被分解成小的变化，通过束搜索过程应用，直到受害者分类器改变其决定。评估证实了我们的方法在受约束的情况下的优越性，特别是在长输入文本（新闻文章）的情况下，其中详尽搜索是不可行的。



## **30. Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks**

黑客攻击人工智能黑客：即时注入作为抵御LLM驱动的网络攻击的防御 cs.CR

v0.1

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.20911v1) [paper-pdf](http://arxiv.org/pdf/2410.20911v1)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: Large language models (LLMs) are increasingly being harnessed to automate cyberattacks, making sophisticated exploits more accessible and scalable. In response, we propose a new defense strategy tailored to counter LLM-driven cyberattacks. We introduce Mantis, a defensive framework that exploits LLMs' susceptibility to adversarial inputs to undermine malicious operations. Upon detecting an automated cyberattack, Mantis plants carefully crafted inputs into system responses, leading the attacker's LLM to disrupt their own operations (passive defense) or even compromise the attacker's machine (active defense). By deploying purposefully vulnerable decoy services to attract the attacker and using dynamic prompt injections for the attacker's LLM, Mantis can autonomously hack back the attacker. In our experiments, Mantis consistently achieved over 95% effectiveness against automated LLM-driven attacks. To foster further research and collaboration, Mantis is available as an open-source tool: https://github.com/pasquini-dario/project_mantis

摘要: 大型语言模型(LLM)越来越多地被用来自动化网络攻击，使复杂的利用更容易获得和可扩展。作为回应，我们提出了一种新的防御战略，以对抗LLM驱动的网络攻击。我们引入了Mantis，这是一个防御框架，利用LLMS对对手输入的敏感性来破坏恶意操作。在检测到自动网络攻击后，螳螂工厂会精心设计输入到系统响应中，导致攻击者的LLM扰乱自己的操作(被动防御)，甚至危害攻击者的机器(主动防御)。通过部署故意易受攻击的诱骗服务来吸引攻击者，并对攻击者的LLM使用动态提示注入，螳螂可以自主地攻击攻击者。在我们的实验中，螳螂对自动LLM驱动的攻击始终取得了95%以上的效率。为了促进进一步的研究和合作，Mantis以开源工具的形式提供：https://github.com/pasquini-dario/project_mantis



## **31. Evaluating the Robustness of LiDAR Point Cloud Tracking Against Adversarial Attack**

评估LiDART点云跟踪对抗攻击的鲁棒性 cs.CV

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.20893v1) [paper-pdf](http://arxiv.org/pdf/2410.20893v1)

**Authors**: Shengjing Tian, Yinan Han, Xiantong Zhao, Bin Liu, Xiuping Liu

**Abstract**: In this study, we delve into the robustness of neural network-based LiDAR point cloud tracking models under adversarial attacks, a critical aspect often overlooked in favor of performance enhancement. These models, despite incorporating advanced architectures like Transformer or Bird's Eye View (BEV), tend to neglect robustness in the face of challenges such as adversarial attacks, domain shifts, or data corruption. We instead focus on the robustness of the tracking models under the threat of adversarial attacks. We begin by establishing a unified framework for conducting adversarial attacks within the context of 3D object tracking, which allows us to thoroughly investigate both white-box and black-box attack strategies. For white-box attacks, we tailor specific loss functions to accommodate various tracking paradigms and extend existing methods such as FGSM, C\&W, and PGD to the point cloud domain. In addressing black-box attack scenarios, we introduce a novel transfer-based approach, the Target-aware Perturbation Generation (TAPG) algorithm, with the dual objectives of achieving high attack performance and maintaining low perceptibility. This method employs a heuristic strategy to enforce sparse attack constraints and utilizes random sub-vector factorization to bolster transferability. Our experimental findings reveal a significant vulnerability in advanced tracking methods when subjected to both black-box and white-box attacks, underscoring the necessity for incorporating robustness against adversarial attacks into the design of LiDAR point cloud tracking models. Notably, compared to existing methods, the TAPG also strikes an optimal balance between the effectiveness of the attack and the concealment of the perturbations.

摘要: 在这项研究中，我们深入研究了基于神经网络的LiDAR点云跟踪模型在对抗攻击下的稳健性，这是一个经常被忽略的关键方面，有助于提高性能。尽管这些模型集成了Transformer或Bird‘s Eye View(BEV)等高级架构，但在面临对手攻击、域转移或数据损坏等挑战时往往忽略了健壮性。相反，我们关注的是跟踪模型在对抗性攻击威胁下的健壮性。我们首先建立一个统一的框架，在3D对象跟踪的背景下进行对抗性攻击，这使我们能够彻底调查白盒和黑盒攻击策略。对于白盒攻击，我们定制了特定的损失函数以适应不同的跟踪范例，并将现有的方法如FGSM、C\&W和PGD扩展到点云域。针对黑盒攻击场景，我们引入了一种新的基于传输的方法，目标感知扰动生成(TAPG)算法，其双重目标是实现高攻击性能和保持低可感知性。该方法使用启发式策略来实施稀疏攻击约束，并利用随机子向量分解来增强可转移性。我们的实验结果揭示了高级跟踪方法在同时受到黑盒和白盒攻击时的显著漏洞，强调了在设计LiDAR点云跟踪模型时考虑对对手攻击的健壮性的必要性。值得注意的是，与现有方法相比，TAPG还在攻击的有效性和扰动的隐蔽性之间取得了最佳平衡。



## **32. GREAT Score: Global Robustness Evaluation of Adversarial Perturbation using Generative Models**

GREAT Score：使用生成模型对对抗性扰动进行全球稳健性评估 cs.LG

10 pages, 13 figures

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2304.09875v3) [paper-pdf](http://arxiv.org/pdf/2304.09875v3)

**Authors**: Zaitang Li, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Current studies on adversarial robustness mainly focus on aggregating local robustness results from a set of data samples to evaluate and rank different models. However, the local statistics may not well represent the true global robustness of the underlying unknown data distribution. To address this challenge, this paper makes the first attempt to present a new framework, called GREAT Score , for global robustness evaluation of adversarial perturbation using generative models. Formally, GREAT Score carries the physical meaning of a global statistic capturing a mean certified attack-proof perturbation level over all samples drawn from a generative model. For finite-sample evaluation, we also derive a probabilistic guarantee on the sample complexity and the difference between the sample mean and the true mean. GREAT Score has several advantages: (1) Robustness evaluations using GREAT Score are efficient and scalable to large models, by sparing the need of running adversarial attacks. In particular, we show high correlation and significantly reduced computation cost of GREAT Score when compared to the attack-based model ranking on RobustBench (Croce,et. al. 2021). (2) The use of generative models facilitates the approximation of the unknown data distribution. In our ablation study with different generative adversarial networks (GANs), we observe consistency between global robustness evaluation and the quality of GANs. (3) GREAT Score can be used for remote auditing of privacy-sensitive black-box models, as demonstrated by our robustness evaluation on several online facial recognition services.

摘要: 目前关于对抗稳健性的研究主要集中在从一组数据样本中聚集局部稳健性结果来评估和排序不同的模型。然而，局部统计可能不能很好地代表潜在未知数据分布的真实全局稳健性。为了应对这一挑战，本文首次尝试提出了一种新的框架，称为Great Score，用于利用产生式模型评估对抗扰动的全局稳健性。在形式上，高分具有全球统计的物理意义，该统计捕获来自生成模型的所有样本的平均经认证的防攻击扰动水平。对于有限样本评价，我们还得到了样本复杂度和样本均值与真均值之差的概率保证。Great Score有几个优点：(1)使用Great Score进行健壮性评估是高效的，并且可以扩展到大型模型，因为它避免了运行对抗性攻击的需要。特别是，与基于攻击的模型排名相比，我们表现出了高度的相关性和显著的降低了计算开销。艾尔2021年)。(2)生成模型的使用有利于未知数据分布的近似。在我们对不同生成对抗网络(GANS)的消融研究中，我们观察到全局健壮性评估与GANS质量之间的一致性。(3)Great Score可以用于隐私敏感的黑盒模型的远程审计，我们在几种在线人脸识别服务上的健壮性评估证明了这一点。



## **33. Robust Text Classification: Analyzing Prototype-Based Networks**

稳健的文本分类：分析基于原型的网络 cs.CL

Published at EMNLP Findings 2024

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2311.06647v3) [paper-pdf](http://arxiv.org/pdf/2311.06647v3)

**Authors**: Zhivar Sourati, Darshan Deshpande, Filip Ilievski, Kiril Gashteovski, Sascha Saralajew

**Abstract**: Downstream applications often require text classification models to be accurate and robust. While the accuracy of the state-of-the-art Language Models (LMs) approximates human performance, they often exhibit a drop in performance on noisy data found in the real world. This lack of robustness can be concerning, as even small perturbations in the text, irrelevant to the target task, can cause classifiers to incorrectly change their predictions. A potential solution can be the family of Prototype-Based Networks (PBNs) that classifies examples based on their similarity to prototypical examples of a class (prototypes) and has been shown to be robust to noise for computer vision tasks. In this paper, we study whether the robustness properties of PBNs transfer to text classification tasks under both targeted and static adversarial attack settings. Our results show that PBNs, as a mere architectural variation of vanilla LMs, offer more robustness compared to vanilla LMs under both targeted and static settings. We showcase how PBNs' interpretability can help us to understand PBNs' robustness properties. Finally, our ablation studies reveal the sensitivity of PBNs' robustness to how strictly clustering is done in the training phase, as tighter clustering results in less robust PBNs.

摘要: 下游应用通常要求文本分类模型准确和健壮。虽然最先进的语言模型(LMS)的准确性接近人类的表现，但它们在处理现实世界中发现的噪声数据时往往表现出性能下降。这种缺乏稳健性可能会令人担忧，因为即使文本中与目标任务无关的微小扰动也可能导致分类器错误地改变他们的预测。一个潜在的解决方案可以是基于原型的网络(PBN)家族，其基于实例与一类(原型)的原型实例的相似性来对实例进行分类，并且已经被证明对计算机视觉任务的噪声是稳健的。本文研究了在目标攻击和静态攻击两种情况下，PBN的健壮性是否会转移到文本分类任务上。我们的结果表明，与普通LMS相比，PBN在目标和静态环境下都提供了更好的健壮性。我们展示了PBN的可解释性如何帮助我们理解PBN的健壮性。最后，我们的消融研究揭示了PBN的稳健性对训练阶段如何严格地进行聚类的敏感性，因为更紧密的聚类会导致更不健壮的PBN。



## **34. Meta-Learning Approaches for Improving Detection of Unseen Speech Deepfakes**

用于改进不可见语音Deepfakes检测的元学习方法 eess.AS

6 pages, accepted to the IEEE Spoken Language Technology Workshop  (SLT) 2024

**SubmitDate**: 2024-10-27    [abs](http://arxiv.org/abs/2410.20578v1) [paper-pdf](http://arxiv.org/pdf/2410.20578v1)

**Authors**: Ivan Kukanov, Janne Laakkonen, Tomi Kinnunen, Ville Hautamäki

**Abstract**: Current speech deepfake detection approaches perform satisfactorily against known adversaries; however, generalization to unseen attacks remains an open challenge. The proliferation of speech deepfakes on social media underscores the need for systems that can generalize to unseen attacks not observed during training. We address this problem from the perspective of meta-learning, aiming to learn attack-invariant features to adapt to unseen attacks with very few samples available. This approach is promising since generating of a high-scale training dataset is often expensive or infeasible. Our experiments demonstrated an improvement in the Equal Error Rate (EER) from 21.67% to 10.42% on the InTheWild dataset, using just 96 samples from the unseen dataset. Continuous few-shot adaptation ensures that the system remains up-to-date.

摘要: 当前的语音深度伪造检测方法对已知对手的表现令人满意;然而，对不可见攻击的概括仍然是一个悬而未决的挑战。社交媒体上语音深度造假的激增凸显了对能够概括训练期间未观察到的不可见攻击的系统的需求。我们从元学习的角度解决这个问题，旨在学习攻击不变的特征，以适应使用很少的样本的不可见的攻击。这种方法很有希望，因为生成大规模训练数据集通常昂贵或不可行。我们的实验表明，仅使用未见过数据集的96个样本，InTheWild数据集的等错误率（EER）从21.67%提高到10.42%。连续的少量镜头调整确保系统保持最新状态。



## **35. LLM Robustness Against Misinformation in Biomedical Question Answering**

LLM生物医学问题回答中针对错误信息的稳健性 cs.CL

**SubmitDate**: 2024-10-27    [abs](http://arxiv.org/abs/2410.21330v1) [paper-pdf](http://arxiv.org/pdf/2410.21330v1)

**Authors**: Alexander Bondarenko, Adrian Viehweger

**Abstract**: The retrieval-augmented generation (RAG) approach is used to reduce the confabulation of large language models (LLMs) for question answering by retrieving and providing additional context coming from external knowledge sources (e.g., by adding the context to the prompt). However, injecting incorrect information can mislead the LLM to generate an incorrect answer.   In this paper, we evaluate the effectiveness and robustness of four LLMs against misinformation - Gemma 2, GPT-4o-mini, Llama~3.1, and Mixtral - in answering biomedical questions. We assess the answer accuracy on yes-no and free-form questions in three scenarios: vanilla LLM answers (no context is provided), "perfect" augmented generation (correct context is provided), and prompt-injection attacks (incorrect context is provided). Our results show that Llama 3.1 (70B parameters) achieves the highest accuracy in both vanilla (0.651) and "perfect" RAG (0.802) scenarios. However, the accuracy gap between the models almost disappears with "perfect" RAG, suggesting its potential to mitigate the LLM's size-related effectiveness differences.   We further evaluate the ability of the LLMs to generate malicious context on one hand and the LLM's robustness against prompt-injection attacks on the other hand, using metrics such as attack success rate (ASR), accuracy under attack, and accuracy drop. As adversaries, we use the same four LLMs (Gemma 2, GPT-4o-mini, Llama 3.1, and Mixtral) to generate incorrect context that is injected in the target model's prompt. Interestingly, Llama is shown to be the most effective adversary, causing accuracy drops of up to 0.48 for vanilla answers and 0.63 for "perfect" RAG across target models. Our analysis reveals that robustness rankings vary depending on the evaluation measure, highlighting the complexity of assessing LLM resilience to adversarial attacks.

摘要: 检索-增强生成(RAG)方法用于通过检索和提供来自外部知识源的附加上下文(例如，通过将上下文添加到提示)来减少大语言模型(LLM)对问题回答的虚构。然而，注入错误的信息可能会误导LLM生成错误的答案。在这篇文章中，我们评估了四个针对错误信息的最小二乘模型-Gema2、GPT-40-mini、Llama~3.1和Mixtral-在回答生物医学问题时的有效性和稳健性。我们在三个场景中评估了是-否和自由形式问题的答案准确率：普通LLM答案(没有提供上下文)、“完美”增强生成(提供了正确的上下文)和提示注入攻击(提供了错误的上下文)。我们的结果表明，Llama3.1(70B参数)在Vanilla(0.651)和“Perfect”RAG(0.802)场景中都达到了最高的精度。然而，随着RAG的“完美”，两个模型之间的精度差距几乎消失了，这表明它有可能缓解LLM与大小相关的有效性差异。我们使用攻击成功率(ASR)、攻击下的准确率和准确率下降等指标，进一步评估了LLM一方面产生恶意上下文的能力，另一方面LLM对即时注入攻击的健壮性。作为对手，我们使用相同的四个LLM(Gema2、GPT-4o-mini、Llama3.1和Mixtral)来生成错误的上下文，该上下文被注入到目标模型的提示符中。有趣的是，骆驼被证明是最有效的对手，在目标模型上，导致普通答案的准确率下降高达0.48，而“完美”RAG的准确率下降0.63。我们的分析表明，健壮性排名根据评估措施的不同而不同，这突显了评估LLM对对手攻击的弹性的复杂性。



## **36. Integrating uncertainty quantification into randomized smoothing based robustness guarantees**

将不确定性量化集成到基于随机平滑的鲁棒性保证中 cs.LG

**SubmitDate**: 2024-10-27    [abs](http://arxiv.org/abs/2410.20432v1) [paper-pdf](http://arxiv.org/pdf/2410.20432v1)

**Authors**: Sina Däubener, Kira Maag, David Krueger, Asja Fischer

**Abstract**: Deep neural networks have proven to be extremely powerful, however, they are also vulnerable to adversarial attacks which can cause hazardous incorrect predictions in safety-critical applications. Certified robustness via randomized smoothing gives a probabilistic guarantee that the smoothed classifier's predictions will not change within an $\ell_2$-ball around a given input. On the other hand (uncertainty) score-based rejection is a technique often applied in practice to defend models against adversarial attacks. In this work, we fuse these two approaches by integrating a classifier that abstains from predicting when uncertainty is high into the certified robustness framework. This allows us to derive two novel robustness guarantees for uncertainty aware classifiers, namely (i) the radius of an $\ell_2$-ball around the input in which the same label is predicted and uncertainty remains low and (ii) the $\ell_2$-radius of a ball in which the predictions will either not change or be uncertain. While the former provides robustness guarantees with respect to attacks aiming at increased uncertainty, the latter informs about the amount of input perturbation necessary to lead the uncertainty aware model into a wrong prediction. Notably, this is on CIFAR10 up to 20.93% larger than for models not allowing for uncertainty based rejection. We demonstrate, that the novel framework allows for a systematic robustness evaluation of different network architectures and uncertainty measures and to identify desired properties of uncertainty quantification techniques. Moreover, we show that leveraging uncertainty in a smoothed classifier helps out-of-distribution detection.

摘要: 深度神经网络已被证明是非常强大的，然而，它们也容易受到对手攻击，这些攻击可能会在安全关键应用中导致危险的错误预测。通过随机平滑证明的稳健性给出了一个概率保证，即平滑后的分类器的预测不会在给定输入周围的$\ell_2$球内改变。另一方面，(不确定性)基于分数的拒绝是一种在实践中经常用于保护模型免受对手攻击的技术。在这项工作中，我们将这两种方法融合在一起，将一个在不确定性较高时不进行预测的分类器集成到经过认证的健壮性框架中。这使得我们可以推导出两个新的不确定性感知分类器的稳健性保证，即(I)预测相同标签且不确定性保持较低的$\ell_2$-球围绕输入的半径，以及(Ii)预测不变或不确定的$\ell_2$-球的半径。前者提供了针对针对增加不确定性的攻击的稳健性保证，而后者通知了将不确定性感知模型引入错误预测所需的输入扰动量。值得注意的是，在CIFAR10上，这比不允许基于不确定性的拒绝的模型高出20.93%。我们证明，新的框架允许对不同的网络体系结构和不确定性度量进行系统的稳健性评估，并识别不确定性量化技术所需的特性。此外，我们还表明，利用平滑分类器中的不确定性有助于离散性检测。



## **37. Classification under strategic adversary manipulation using pessimistic bilevel optimisation**

使用悲观二层优化进行战略对手操纵下的分类 cs.LG

27 pages, 5 figures, under review

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2410.20284v1) [paper-pdf](http://arxiv.org/pdf/2410.20284v1)

**Authors**: David Benfield, Stefano Coniglio, Martin Kunc, Phan Tu Vuong, Alain Zemkoho

**Abstract**: Adversarial machine learning concerns situations in which learners face attacks from active adversaries. Such scenarios arise in applications such as spam email filtering, malware detection and fake-image generation, where security methods must be actively updated to keep up with the ever improving generation of malicious data.We model these interactions between the learner and the adversary as a game and formulate the problem as a pessimistic bilevel optimisation problem with the learner taking the role of the leader. The adversary, modelled as a stochastic data generator, takes the role of the follower, generating data in response to the classifier. While existing models rely on the assumption that the adversary will choose the least costly solution leading to a convex lower-level problem with a unique solution, we present a novel model and solution method which do not make such assumptions. We compare these to the existing approach and see significant improvements in performance suggesting that relaxing these assumptions leads to a more realistic model.

摘要: 对抗性机器学习关注学习者面临活跃对手攻击的情况。这种情况出现在垃圾邮件过滤、恶意软件检测和虚假图像生成等应用中，这些应用中的安全方法必须主动更新，以跟上不断改进的恶意数据的生成。我们将学习者和对手之间的这些交互建模为博弈，并将问题描述为一个悲观的双层优化问题，学习者扮演领导者的角色。敌手被建模为随机数据生成器，扮演跟随者的角色，生成响应分类器的数据。现有的模型依赖于假设对手将选择代价最低的解，从而导致具有唯一解的凸下层问题，而我们提出了一种新的模型和求解方法，该模型和求解方法没有做出这样的假设。我们将这些方法与现有的方法进行了比较，发现性能有了显着的改善，这表明放松这些假设会产生一个更现实的模型。



## **38. CodePurify: Defend Backdoor Attacks on Neural Code Models via Entropy-based Purification**

CodePuriify：通过基于熵的净化防御对神经代码模型的后门攻击 cs.CR

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2410.20136v1) [paper-pdf](http://arxiv.org/pdf/2410.20136v1)

**Authors**: Fangwen Mu, Junjie Wang, Zhuohao Yu, Lin Shi, Song Wang, Mingyang Li, Qing Wang

**Abstract**: Neural code models have found widespread success in tasks pertaining to code intelligence, yet they are vulnerable to backdoor attacks, where an adversary can manipulate the victim model's behavior by inserting triggers into the source code. Recent studies indicate that advanced backdoor attacks can achieve nearly 100% attack success rates on many software engineering tasks. However, effective defense techniques against such attacks remain insufficiently explored. In this study, we propose CodePurify, a novel defense against backdoor attacks on code models through entropy-based purification. Entropy-based purification involves the process of precisely detecting and eliminating the possible triggers in the source code while preserving its semantic information. Within this process, CodePurify first develops a confidence-driven entropy-based measurement to determine whether a code snippet is poisoned and, if so, locates the triggers. Subsequently, it purifies the code by substituting the triggers with benign tokens using a masked language model. We extensively evaluate CodePurify against four advanced backdoor attacks across three representative tasks and two popular code models. The results show that CodePurify significantly outperforms four commonly used defense baselines, improving average defense performance by at least 40%, 40%, and 12% across the three tasks, respectively. These findings highlight the potential of CodePurify to serve as a robust defense against backdoor attacks on neural code models.

摘要: 神经代码模型在与代码智能相关的任务中取得了广泛的成功，但它们容易受到后门攻击，在后门攻击中，对手可以通过在源代码中插入触发器来操纵受害者模型的行为。最近的研究表明，高级后门攻击可以在许多软件工程任务上实现近100%的攻击成功率。然而，针对此类攻击的有效防御技术仍然没有得到充分的探索。在这项研究中，我们提出了CodePurify，一种通过基于熵的净化来防御代码模型后门攻击的新方法。基于熵的净化涉及在保留源代码语义信息的同时精确检测和消除源代码中可能的触发器的过程。在这个过程中，CodePurify首先开发基于置信度的基于熵的度量，以确定代码片段是否有毒，如果是，则定位触发器。随后，它通过使用屏蔽语言模型将触发器替换为良性令牌来净化代码。我们通过三个有代表性的任务和两个流行的代码模型，针对四种高级后门攻击对CodePurify进行了广泛的评估。结果表明，CodePurify显著优于四个常用的防御基线，在三个任务中分别将平均防御性能提高了至少40%、40%和12%。这些发现突出了CodePurify作为对神经代码模型的后门攻击的强大防御的潜力。



## **39. Adversarial Attacks Against Double RIS-Assisted MIMO Systems-based Autoencoder in Finite-Scattering Environments**

伪随机散射环境中针对基于双RIS辅助MMO系统的自动编码器的对抗攻击 cs.IT

5 pages, 2 figures. Accepted by WCL

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2410.20103v1) [paper-pdf](http://arxiv.org/pdf/2410.20103v1)

**Authors**: Bui Duc Son, Ngo Nam Khanh, Trinh Van Chien, Dong In Kim

**Abstract**: Autoencoder permits the end-to-end optimization and design of wireless communication systems to be more beneficial than traditional signal processing. However, this emerging learning-based framework has weaknesses, especially sensitivity to physical attacks. This paper explores adversarial attacks against a double reconfigurable intelligent surface (RIS)-assisted multiple-input and multiple-output (MIMO)-based autoencoder, where an adversary employs encoded and decoded datasets to create adversarial perturbation and fool the system. Because of the complex and dynamic data structures, adversarial attacks are not unique, each having its own benefits. We, therefore, propose three algorithms generating adversarial examples and perturbations to attack the RIS-MIMO-based autoencoder, exploiting the gradient descent and allowing for flexibility via varying the input dimensions. Numerical results show that the proposed adversarial attack-based algorithm significantly degrades the system performance regarding the symbol error rate compared to the jamming attacks.

摘要: 自动编码器使无线通信系统的端到端优化和设计比传统的信号处理更有利。然而，这个新兴的基于学习的框架也有弱点，特别是对物理攻击的敏感性。研究了针对双重可重构智能表面(RIS)辅助的多输入多输出(MIMO)自动编码器的敌意攻击，其中敌手利用编码和解码的数据集来制造对抗性扰动并愚弄系统。由于复杂和动态的数据结构，对抗性攻击并不是唯一的，每种攻击都有自己的好处。因此，我们提出了三种生成对抗性示例和扰动的算法来攻击基于RIS-MIMO的自动编码器，利用梯度下降并允许通过改变输入维度来实现灵活性。数值结果表明，与干扰攻击相比，本文提出的基于对抗攻击的算法显著降低了系统的误码率性能。



## **40. Generative Adversarial Patches for Physical Attacks on Cross-Modal Pedestrian Re-Identification**

跨模式行人重新识别物理攻击的生成对抗补丁 cs.CV

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2410.20097v1) [paper-pdf](http://arxiv.org/pdf/2410.20097v1)

**Authors**: Yue Su, Hao Li, Maoguo Gong

**Abstract**: Visible-infrared pedestrian Re-identification (VI-ReID) aims to match pedestrian images captured by infrared cameras and visible cameras. However, VI-ReID, like other traditional cross-modal image matching tasks, poses significant challenges due to its human-centered nature. This is evidenced by the shortcomings of existing methods, which struggle to extract common features across modalities, while losing valuable information when bridging the gap between them in the implicit feature space, potentially compromising security. To address this vulnerability, this paper introduces the first physical adversarial attack against VI-ReID models. Our method, termed Edge-Attack, specifically tests the models' ability to leverage deep-level implicit features by focusing on edge information, the most salient explicit feature differentiating individuals across modalities. Edge-Attack utilizes a novel two-step approach. First, a multi-level edge feature extractor is trained in a self-supervised manner to capture discriminative edge representations for each individual. Second, a generative model based on Vision Transformer Generative Adversarial Networks (ViTGAN) is employed to generate adversarial patches conditioned on the extracted edge features. By applying these patches to pedestrian clothing, we create realistic, physically-realizable adversarial samples. This black-box, self-supervised approach ensures the generalizability of our attack against various VI-ReID models. Extensive experiments on SYSU-MM01 and RegDB datasets, including real-world deployments, demonstrate the effectiveness of Edge- Attack in significantly degrading the performance of state-of-the-art VI-ReID methods.

摘要: 可见光-红外行人再识别(VI-REID)的目标是匹配红外摄像机和可见光摄像机拍摄的行人图像。然而，VI-Reid像其他传统的跨模式图像匹配任务一样，由于其以人为中心的性质而构成了巨大的挑战。现有方法的缺点证明了这一点，这些方法难以跨模式提取共同特征，而当在隐式特征空间中弥合它们之间的差距时，会丢失有价值的信息，这可能会损害安全性。为了解决这一漏洞，本文引入了对VI-Reid模型的第一次物理攻击。我们的方法被称为边缘攻击，通过关注边缘信息来具体测试模型利用深层隐式特征的能力，边缘信息是最显著的显式特征，可以区分不同模式的个体。边缘攻击采用了一种新的两步法。首先，以自监督方式训练多层边缘特征提取器，以捕获每个个体的区别性边缘表示。其次，利用基于视觉变换生成对抗性网络(ViTGAN)的生成模型，根据提取的边缘特征生成对抗性斑块。通过将这些补丁应用到行人服装上，我们创建了逼真的、物理上可实现的对抗性样本。这种黑盒、自我监督的方法确保了我们对各种VI-Reid模型的攻击的通用性。在SYSU-MM01和RegDB数据集上的广泛实验，包括真实世界的部署，证明了边缘攻击在显著降低最先进的VI-Reid方法的性能方面的有效性。



## **41. Attacks against Abstractive Text Summarization Models through Lead Bias and Influence Functions**

通过铅偏差和影响函数对抽象文本摘要模型的攻击 cs.CL

10 pages, 3 figures, Accepted at EMNLP Findings 2024

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2410.20019v1) [paper-pdf](http://arxiv.org/pdf/2410.20019v1)

**Authors**: Poojitha Thota, Shirin Nilizadeh

**Abstract**: Large Language Models have introduced novel opportunities for text comprehension and generation. Yet, they are vulnerable to adversarial perturbations and data poisoning attacks, particularly in tasks like text classification and translation. However, the adversarial robustness of abstractive text summarization models remains less explored. In this work, we unveil a novel approach by exploiting the inherent lead bias in summarization models, to perform adversarial perturbations. Furthermore, we introduce an innovative application of influence functions, to execute data poisoning, which compromises the model's integrity. This approach not only shows a skew in the models behavior to produce desired outcomes but also shows a new behavioral change, where models under attack tend to generate extractive summaries rather than abstractive summaries.

摘要: 大型语言模型为文本理解和生成带来了新的机会。然而，它们很容易受到敌对干扰和数据中毒攻击，特别是在文本分类和翻译等任务中。然而，抽象文本摘要模型的对抗稳健性仍然较少被探索。在这项工作中，我们推出了一种新颖的方法，通过利用摘要模型中固有的领先偏差来执行对抗性扰动。此外，我们引入了影响函数的创新应用程序，以执行数据中毒，这会损害模型的完整性。这种方法不仅显示了模型产生所需结果的行为的倾斜，而且还显示了新的行为变化，即受攻击的模型倾向于生成提取摘要而不是抽象摘要。



## **42. Backdoor in Seconds: Unlocking Vulnerabilities in Large Pre-trained Models via Model Editing**

秒内后门：通过模型编辑解锁大型预训练模型中的漏洞 cs.AI

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.18267v2) [paper-pdf](http://arxiv.org/pdf/2410.18267v2)

**Authors**: Dongliang Guo, Mengxuan Hu, Zihan Guan, Junfeng Guo, Thomas Hartvigsen, Sheng Li

**Abstract**: Large pre-trained models have achieved notable success across a range of downstream tasks. However, recent research shows that a type of adversarial attack ($\textit{i.e.,}$ backdoor attack) can manipulate the behavior of machine learning models through contaminating their training dataset, posing significant threat in the real-world application of large pre-trained model, especially for those customized models. Therefore, addressing the unique challenges for exploring vulnerability of pre-trained models is of paramount importance. Through empirical studies on the capability for performing backdoor attack in large pre-trained models ($\textit{e.g.,}$ ViT), we find the following unique challenges of attacking large pre-trained models: 1) the inability to manipulate or even access large training datasets, and 2) the substantial computational resources required for training or fine-tuning these models. To address these challenges, we establish new standards for an effective and feasible backdoor attack in the context of large pre-trained models. In line with these standards, we introduce our EDT model, an \textbf{E}fficient, \textbf{D}ata-free, \textbf{T}raining-free backdoor attack method. Inspired by model editing techniques, EDT injects an editing-based lightweight codebook into the backdoor of large pre-trained models, which replaces the embedding of the poisoned image with the target image without poisoning the training dataset or training the victim model. Our experiments, conducted across various pre-trained models such as ViT, CLIP, BLIP, and stable diffusion, and on downstream tasks including image classification, image captioning, and image generation, demonstrate the effectiveness of our method. Our code is available in the supplementary material.

摘要: 大型预先培训的模型在一系列下游任务中取得了显着的成功。然而，最近的研究表明，一种对抗性攻击(即后门攻击)可以通过污染机器学习模型的训练数据集来操纵它们的行为，这对大型预训练模型的实际应用构成了巨大的威胁，特别是对那些定制的模型。因此，解决探索预先训练模型的脆弱性的独特挑战是至关重要的。通过对大型预训练模型执行后门攻击能力的实证研究，我们发现攻击大型预训练模型面临以下独特的挑战：1)无法操纵甚至访问大型训练数据集；2)训练或微调这些模型所需的大量计算资源。为了应对这些挑战，我们在大型预训练模型的背景下为有效和可行的后门攻击建立了新的标准。根据这些标准，我们介绍了我们的EDT模型，一种高效、无ATA、无雨的后门攻击方法。受模型编辑技术的启发，EDT将基于编辑的轻量级码本注入大型预训练模型的后门，在不毒化训练数据集或训练受害者模型的情况下，将有毒图像的嵌入替换为目标图像。我们在VIT、CLIP、BIP和稳定扩散等各种预先训练的模型上进行的实验，以及在图像分类、图像字幕和图像生成等下游任务上进行的实验，证明了该方法的有效性。我们的代码可以在补充材料中找到。



## **43. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

迭代自调优LLM以增强越狱能力 cs.CL

18 pages

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.18469v2) [paper-pdf](http://arxiv.org/pdf/2410.18469v2)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99% ASR on GPT-3.5 and 49% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety.

摘要: 最近的研究表明，大型语言模型(LLM)容易受到自动越狱攻击，在自动越狱攻击中，由附加到有害查询的算法编制的敌意后缀绕过安全对齐并触发意外响应。目前生成这些后缀的方法计算量大，攻击成功率(ASR)低，尤其是针对Llama2和Llama3等排列良好的模型。为了克服这些限制，我们引入了ADV-LLM，这是一个迭代的自我调整过程，可以制作具有增强越狱能力的对抗性LLM。我们的框架大大降低了生成敌意后缀的计算代价，同时在各种开源LLM上实现了近100个ASR。此外，它表现出很强的攻击可转换性，尽管只在Llama3上进行了优化，但在GPT-3.5上实现了99%的ASR，在GPT-4上实现了49%的ASR。除了提高越狱能力，ADV-LLM还通过其生成用于研究LLM安全性的大型数据集的能力，为未来的安全配准研究提供了有价值的见解。



## **44. RobustKV: Defending Large Language Models against Jailbreak Attacks via KV Eviction**

RobustKN：通过GV驱逐保护大型语言模型免受越狱攻击 cs.CR

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.19937v1) [paper-pdf](http://arxiv.org/pdf/2410.19937v1)

**Authors**: Tanqiu Jiang, Zian Wang, Jiacheng Liang, Changjiang Li, Yuhui Wang, Ting Wang

**Abstract**: Jailbreak attacks circumvent LLMs' built-in safeguards by concealing harmful queries within jailbreak prompts. While existing defenses primarily focus on mitigating the effects of jailbreak prompts, they often prove inadequate as jailbreak prompts can take arbitrary, adaptive forms. This paper presents RobustKV, a novel defense that adopts a fundamentally different approach by selectively removing critical tokens of harmful queries from key-value (KV) caches. Intuitively, for a jailbreak prompt to be effective, its tokens must achieve sufficient `importance' (as measured by attention scores), which inevitably lowers the importance of tokens in the concealed harmful query. Thus, by strategically evicting the KVs of the lowest-ranked tokens, RobustKV diminishes the presence of the harmful query in the KV cache, thus preventing the LLM from generating malicious responses. Extensive evaluation using benchmark datasets and models demonstrates that RobustKV effectively counters state-of-the-art jailbreak attacks while maintaining the LLM's general performance on benign queries. Moreover, RobustKV creates an intriguing evasiveness dilemma for adversaries, forcing them to balance between evading RobustKV and bypassing the LLM's built-in safeguards. This trade-off contributes to RobustKV's robustness against adaptive attacks. (warning: this paper contains potentially harmful content generated by LLMs.)

摘要: 越狱攻击通过在越狱提示中隐藏有害的查询来绕过LLMS的内置保护措施。虽然现有的防御措施主要集中在减轻越狱提示的影响，但它们往往被证明是不够的，因为越狱提示可以采取任意的、适应性的形式。本文提出了一种新的防御方法RobustKV，它采用了一种从根本上不同的方法，从键值(KV)缓存中选择性地删除有害查询的关键标记。直观地说，要使越狱提示有效，其标记必须达到足够的“重要性”(通过注意力得分来衡量)，这不可避免地降低了标记在隐藏的有害查询中的重要性。因此，通过战略性地逐出最低等级令牌的KV，RobustKV减少了KV缓存中有害查询的存在，从而防止LLM生成恶意响应。使用基准数据集和模型进行的广泛评估表明，RobustKV在保持LLM在良性查询上的总体性能的同时，有效地对抗了最先进的越狱攻击。此外，RobustKV为对手创造了一个耐人寻味的逃避困境，迫使他们在躲避RobustKV和绕过LLM的内置保障之间取得平衡。这种权衡有助于RobustKV对自适应攻击的健壮性。(警告：本文包含由LLMS生成的潜在有害内容。)



## **45. Robust Thompson Sampling Algorithms Against Reward Poisoning Attacks**

针对奖励中毒攻击的稳健Thompson抽样算法 cs.LG

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.19705v1) [paper-pdf](http://arxiv.org/pdf/2410.19705v1)

**Authors**: Yinglun Xu, Zhiwei Wang, Gagandeep Singh

**Abstract**: Thompson sampling is one of the most popular learning algorithms for online sequential decision-making problems and has rich real-world applications. However, current Thompson sampling algorithms are limited by the assumption that the rewards received are uncorrupted, which may not be true in real-world applications where adversarial reward poisoning exists. To make Thompson sampling more reliable, we want to make it robust against adversarial reward poisoning. The main challenge is that one can no longer compute the actual posteriors for the true reward, as the agent can only observe the rewards after corruption. In this work, we solve this problem by computing pseudo-posteriors that are less likely to be manipulated by the attack. We propose robust algorithms based on Thompson sampling for the popular stochastic and contextual linear bandit settings in both cases where the agent is aware or unaware of the budget of the attacker. We theoretically show that our algorithms guarantee near-optimal regret under any attack strategy.

摘要: Thompson抽样是在线序贯决策问题中最常用的学习算法之一，在现实世界中有着广泛的应用。然而，当前的Thompson采样算法受到接收到的奖励是未被破坏的假设的限制，这在存在对抗性奖励中毒的现实应用中可能是不成立的。为了使汤普森抽样更可靠，我们希望使其对对手奖励中毒具有健壮性。主要的挑战是，人们不再能计算出真正报酬的实际后遗症，因为代理人只能观察腐败后的报酬。在这项工作中，我们通过计算不太可能被攻击操纵的伪后验来解决这个问题。对于常见的随机和上下文线性盗贼设置，我们提出了基于Thompson采样的稳健算法，在代理知道和不知道攻击者预算的两种情况下都是如此。我们从理论上证明了我们的算法在任何攻击策略下都能保证近似最优的错误。



## **46. A constrained optimization approach to improve robustness of neural networks**

提高神经网络鲁棒性的约束优化方法 cs.LG

29 pages, 4 figures, 5 tables

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2409.13770v2) [paper-pdf](http://arxiv.org/pdf/2409.13770v2)

**Authors**: Shudian Zhao, Jan Kronqvist

**Abstract**: In this paper, we present a novel nonlinear programming-based approach to fine-tune pre-trained neural networks to improve robustness against adversarial attacks while maintaining high accuracy on clean data. Our method introduces adversary-correction constraints to ensure correct classification of adversarial data and minimizes changes to the model parameters. We propose an efficient cutting-plane-based algorithm to iteratively solve the large-scale nonconvex optimization problem by approximating the feasible region through polyhedral cuts and balancing between robustness and accuracy. Computational experiments on standard datasets such as MNIST and CIFAR10 demonstrate that the proposed approach significantly improves robustness, even with a very small set of adversarial data, while maintaining minimal impact on accuracy.

摘要: 在本文中，我们提出了一种新型的基于非线性规划的方法来微调预训练的神经网络，以提高针对对抗性攻击的鲁棒性，同时保持干净数据的高准确性。我们的方法引入了对抗修正约束，以确保对抗数据的正确分类，并最大限度地减少对模型参数的更改。我们提出了一种高效的基于切割平面的算法，通过通过多边形切割逼近可行区域并平衡鲁棒性和准确性来迭代解决大规模非凸优化问题。对MNIST和CIFAR 10等标准数据集的计算实验表明，即使使用非常小的对抗数据集，所提出的方法也能显着提高稳健性，同时保持对准确性的影响最小。



## **47. Detecting adversarial attacks on random samples**

检测对随机样本的对抗攻击 math.PR

title changed; introduction expanded; new results about spherical  attacks

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2408.06166v2) [paper-pdf](http://arxiv.org/pdf/2408.06166v2)

**Authors**: Gleb Smirnov

**Abstract**: This paper studies the problem of detecting adversarial perturbations in a sequence of observations. Given a data sample $X_1, \ldots, X_n$ drawn from a standard normal distribution, an adversary, after observing the sample, can perturb each observation by a fixed magnitude or leave it unchanged. We explore the relationship between the perturbation magnitude, the sparsity of the perturbation, and the detectability of the adversary's actions, establishing precise thresholds for when detection becomes impossible.

摘要: 本文研究了在观察序列中检测对抗性扰动的问题。给定从标准正态分布中提取的数据样本$X_1，\ldots，X_n$，对手在观察样本后可以以固定幅度扰乱每个观察或保持其不变。我们探索了扰动幅度、扰动的稀疏性和对手行为的可检测性之间的关系，为何时检测变得不可能建立精确的阈值。



## **48. Corpus Poisoning via Approximate Greedy Gradient Descent**

通过近似贪婪梯度下降来中毒 cs.IR

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2406.05087v2) [paper-pdf](http://arxiv.org/pdf/2406.05087v2)

**Authors**: Jinyan Su, Preslav Nakov, Claire Cardie

**Abstract**: Dense retrievers are widely used in information retrieval and have also been successfully extended to other knowledge intensive areas such as language models, e.g., Retrieval-Augmented Generation (RAG) systems. Unfortunately, they have recently been shown to be vulnerable to corpus poisoning attacks in which a malicious user injects a small fraction of adversarial passages into the retrieval corpus to trick the system into returning these passages among the top-ranked results for a broad set of user queries. Further study is needed to understand the extent to which these attacks could limit the deployment of dense retrievers in real-world applications. In this work, we propose Approximate Greedy Gradient Descent (AGGD), a new attack on dense retrieval systems based on the widely used HotFlip method for efficiently generating adversarial passages. We demonstrate that AGGD can select a higher quality set of token-level perturbations than HotFlip by replacing its random token sampling with a more structured search. Experimentally, we show that our method achieves a high attack success rate on several datasets and using several retrievers, and can generalize to unseen queries and new domains. Notably, our method is extremely effective in attacking the ANCE retrieval model, achieving attack success rates that are 15.24\% and 17.44\% higher on the NQ and MS MARCO datasets, respectively, compared to HotFlip. Additionally, we demonstrate AGGD's potential to replace HotFlip in other adversarial attacks, such as knowledge poisoning of RAG systems.

摘要: 密集检索器被广泛应用于信息检索，也被成功地扩展到其他知识密集型领域，例如语言模型，例如检索-增强生成(RAG)系统。不幸的是，它们最近被证明容易受到语料库中毒攻击，在这种攻击中，恶意用户将一小部分对抗性段落注入检索语料库，以欺骗系统返回针对广泛的用户查询集合的排名靠前的结果中的这些段落。需要进一步的研究来了解这些攻击在多大程度上会限制密集检索器在现实世界应用中的部署。在这项工作中，我们提出了近似贪婪梯度下降(AGGD)，一种新的攻击密集检索系统的基础上，广泛使用的HotFlip方法，以有效地生成敌意段落。我们证明，通过用更结构化的搜索取代随机令牌抽样，AGGD可以选择比HotFlip更高质量的令牌级扰动集。实验表明，我们的方法在多个数据集和多个检索器上取得了很高的攻击成功率，并且可以推广到未知的查询和新的领域。值得注意的是，我们的方法在攻击ANCE检索模型方面非常有效，在NQ和MS Marco数据集上的攻击成功率分别比HotFlip高15.24和17.44。此外，我们还展示了AGGD在其他对抗性攻击中取代HotFlip的潜力，例如RAG系统的知识中毒。



## **49. Adversarial Attacks on Large Language Models Using Regularized Relaxation**

使用正规松弛对大型语言模型的对抗攻击 cs.LG

8 pages, 6 figures

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.19160v1) [paper-pdf](http://arxiv.org/pdf/2410.19160v1)

**Authors**: Samuel Jacob Chacko, Sajib Biswas, Chashi Mahiul Islam, Fatema Tabassum Liza, Xiuwen Liu

**Abstract**: As powerful Large Language Models (LLMs) are now widely used for numerous practical applications, their safety is of critical importance. While alignment techniques have significantly improved overall safety, LLMs remain vulnerable to carefully crafted adversarial inputs. Consequently, adversarial attack methods are extensively used to study and understand these vulnerabilities. However, current attack methods face significant limitations. Those relying on optimizing discrete tokens suffer from limited efficiency, while continuous optimization techniques fail to generate valid tokens from the model's vocabulary, rendering them impractical for real-world applications. In this paper, we propose a novel technique for adversarial attacks that overcomes these limitations by leveraging regularized gradients with continuous optimization methods. Our approach is two orders of magnitude faster than the state-of-the-art greedy coordinate gradient-based method, significantly improving the attack success rate on aligned language models. Moreover, it generates valid tokens, addressing a fundamental limitation of existing continuous optimization methods. We demonstrate the effectiveness of our attack on five state-of-the-art LLMs using four datasets.

摘要: 随着强大的大型语言模型(LLM)在众多实际应用中的广泛应用，它们的安全性至关重要。虽然对齐技术显著提高了总体安全性，但LLM仍然容易受到精心设计的敌方输入的影响。因此，对抗性攻击方法被广泛用于研究和理解这些漏洞。然而，目前的攻击方法面临着很大的局限性。那些依赖于优化离散令牌的人效率有限，而连续优化技术无法从模型的词汇表中生成有效的令牌，这使得它们在现实世界中的应用不切实际。在本文中，我们提出了一种新的对抗性攻击技术，通过利用正则化的梯度和连续优化方法来克服这些局限性。我们的方法比最先进的贪婪坐标梯度方法快两个数量级，显著提高了对齐语言模型的攻击成功率。此外，它还生成有效的令牌，解决了现有连续优化方法的一个基本限制。我们使用四个数据集演示了我们对五个最先进的LLM的攻击的有效性。



## **50. Provably Robust Watermarks for Open-Source Language Models**

开源语言模型的可证明稳健的水印 cs.CR

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18861v1) [paper-pdf](http://arxiv.org/pdf/2410.18861v1)

**Authors**: Miranda Christ, Sam Gunn, Tal Malkin, Mariana Raykova

**Abstract**: The recent explosion of high-quality language models has necessitated new methods for identifying AI-generated text. Watermarking is a leading solution and could prove to be an essential tool in the age of generative AI. Existing approaches embed watermarks at inference and crucially rely on the large language model (LLM) specification and parameters being secret, which makes them inapplicable to the open-source setting. In this work, we introduce the first watermarking scheme for open-source LLMs. Our scheme works by modifying the parameters of the model, but the watermark can be detected from just the outputs of the model. Perhaps surprisingly, we prove that our watermarks are unremovable under certain assumptions about the adversary's knowledge. To demonstrate the behavior of our construction under concrete parameter instantiations, we present experimental results with OPT-6.7B and OPT-1.3B. We demonstrate robustness to both token substitution and perturbation of the model parameters. We find that the stronger of these attacks, the model-perturbation attack, requires deteriorating the quality score to 0 out of 100 in order to bring the detection rate down to 50%.

摘要: 最近高质量语言模型的爆炸性增长需要新的方法来识别人工智能生成的文本。水印是一种领先的解决方案，可能会被证明是生成性人工智能时代的重要工具。现有的方法在推理时嵌入水印，重要的是依赖于大型语言模型(LLM)规范和参数是保密的，这使得它们不适用于开源环境。在这项工作中，我们介绍了第一个用于开源LLMS的水印方案。我们的方案通过修改模型的参数来工作，但仅从模型的输出就可以检测到水印。也许令人惊讶的是，我们证明了我们的水印在关于对手知识的某些假设下是不可移除的。为了演示我们的构造在混凝土参数实例化下的行为，我们给出了使用OPT-6.7B和OPT-1.3B的实验结果。我们证明了对令牌替换和模型参数摄动的稳健性。我们发现，在这些攻击中，较强的模型扰动攻击需要将质量分数恶化到0分(满分100分)，才能将检测率降至50%。



