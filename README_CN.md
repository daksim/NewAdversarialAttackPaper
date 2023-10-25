# Latest Adversarial Attack Papers
**update at 2023-10-25 10:04:50**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Diffusion-Based Adversarial Purification for Speaker Verification**

基于扩散的对抗性净化说话人确认 eess.AS

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.14270v2) [paper-pdf](http://arxiv.org/pdf/2310.14270v2)

**Authors**: Yibo Bai, Xiao-Lei Zhang

**Abstract**: Recently, automatic speaker verification (ASV) based on deep learning is easily contaminated by adversarial attacks, which is a new type of attack that injects imperceptible perturbations to audio signals so as to make ASV produce wrong decisions. This poses a significant threat to the security and reliability of ASV systems. To address this issue, we propose a Diffusion-Based Adversarial Purification (DAP) method that enhances the robustness of ASV systems against such adversarial attacks. Our method leverages a conditional denoising diffusion probabilistic model to effectively purify the adversarial examples and mitigate the impact of perturbations. DAP first introduces controlled noise into adversarial examples, and then performs a reverse denoising process to reconstruct clean audio. Experimental results demonstrate the efficacy of the proposed DAP in enhancing the security of ASV and meanwhile minimizing the distortion of the purified audio signals.

摘要: 近年来，基于深度学习的自动说话人确认（ASV）容易受到对抗性攻击的污染，对抗性攻击是一种新型的攻击方式，通过对音频信号注入不可感知的扰动，使ASV产生错误的决策。这对ASV系统的安全性和可靠性构成了重大威胁。为了解决这个问题，我们提出了一种基于扩散的对抗性净化（DAP）方法，该方法增强了ASV系统对这种对抗性攻击的鲁棒性。我们的方法利用条件去噪扩散概率模型来有效地净化对抗性示例并减轻扰动的影响。DAP首先将受控噪声引入对抗性示例，然后执行反向去噪过程以重建干净的音频。实验结果表明，提出的DAP在提高ASV的安全性，同时最大限度地减少净化音频信号的失真的有效性。



## **2. A Survey on LLM-generated Text Detection: Necessity, Methods, and Future Directions**

LLM生成的文本检测：必要性、方法和未来发展方向综述 cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.14724v2) [paper-pdf](http://arxiv.org/pdf/2310.14724v2)

**Authors**: Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Derek F. Wong, Lidia S. Chao

**Abstract**: The powerful ability to understand, follow, and generate complex language emerging from large language models (LLMs) makes LLM-generated text flood many areas of our daily lives at an incredible speed and is widely accepted by humans. As LLMs continue to expand, there is an imperative need to develop detectors that can detect LLM-generated text. This is crucial to mitigate potential misuse of LLMs and safeguard realms like artistic expression and social networks from harmful influence of LLM-generated content. The LLM-generated text detection aims to discern if a piece of text was produced by an LLM, which is essentially a binary classification task. The detector techniques have witnessed notable advancements recently, propelled by innovations in watermarking techniques, zero-shot methods, fine-turning LMs methods, adversarial learning methods, LLMs as detectors, and human-assisted methods. In this survey, we collate recent research breakthroughs in this area and underscore the pressing need to bolster detector research. We also delve into prevalent datasets, elucidating their limitations and developmental requirements. Furthermore, we analyze various LLM-generated text detection paradigms, shedding light on challenges like out-of-distribution problems, potential attacks, and data ambiguity. Conclusively, we highlight interesting directions for future research in LLM-generated text detection to advance the implementation of responsible artificial intelligence (AI). Our aim with this survey is to provide a clear and comprehensive introduction for newcomers while also offering seasoned researchers a valuable update in the field of LLM-generated text detection. The useful resources are publicly available at: https://github.com/NLP2CT/LLM-generated-Text-Detection.

摘要: 大型语言模型(LLM)强大的理解、跟踪和生成复杂语言的能力使得LLM生成的文本以令人难以置信的速度涌入我们日常生活的许多领域，并被人类广泛接受。随着LLMS的不断扩展，迫切需要开发能够检测LLM生成的文本的检测器。这对于减少LLM的潜在滥用以及保护艺术表达和社交网络等领域免受LLM生成的内容的有害影响至关重要。LLM生成的文本检测旨在识别一段文本是否由LLM生成，这本质上是一项二进制分类任务。最近，在水印技术、零镜头方法、精细旋转LMS方法、对抗性学习方法、作为检测器的LLMS以及人工辅助方法的创新的推动下，检测器技术有了显著的进步。在这次调查中，我们整理了这一领域的最新研究突破，并强调了支持探测器研究的迫切需要。我们还深入研究了流行的数据集，阐明了它们的局限性和发展需求。此外，我们分析了各种LLM生成的文本检测范例，揭示了诸如分发外问题、潜在攻击和数据歧义等挑战。最后，我们指出了未来在LLM生成的文本检测方面的有趣研究方向，以推进负责任人工智能(AI)的实施。我们这次调查的目的是为新手提供一个清晰而全面的介绍，同时也为经验丰富的研究人员提供在LLM生成的文本检测领域的有价值的更新。这些有用的资源可在以下网址公开获得：https://github.com/NLP2CT/LLM-generated-Text-Detection.



## **3. Momentum Gradient-based Untargeted Attack on Hypergraph Neural Networks**

基于动量梯度的超图神经网络无目标攻击 cs.LG

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15656v1) [paper-pdf](http://arxiv.org/pdf/2310.15656v1)

**Authors**: Yang Chen, Stjepan Picek, Zhonglin Ye, Zhaoyang Wang, Haixing Zhao

**Abstract**: Hypergraph Neural Networks (HGNNs) have been successfully applied in various hypergraph-related tasks due to their excellent higher-order representation capabilities. Recent works have shown that deep learning models are vulnerable to adversarial attacks. Most studies on graph adversarial attacks have focused on Graph Neural Networks (GNNs), and the study of adversarial attacks on HGNNs remains largely unexplored. In this paper, we try to reduce this gap. We design a new HGNNs attack model for the untargeted attack, namely MGHGA, which focuses on modifying node features. We consider the process of HGNNs training and use a surrogate model to implement the attack before hypergraph modeling. Specifically, MGHGA consists of two parts: feature selection and feature modification. We use a momentum gradient mechanism to choose the attack node features in the feature selection module. In the feature modification module, we use two feature generation approaches (direct modification and sign gradient) to enable MGHGA to be employed on discrete and continuous datasets. We conduct extensive experiments on five benchmark datasets to validate the attack performance of MGHGA in the node and the visual object classification tasks. The results show that MGHGA improves performance by an average of 2% compared to the than the baselines.

摘要: 超图神经网络(HGNN)因其优良的高阶表示能力而被成功地应用于各种与超图相关的任务中。最近的研究表明，深度学习模型容易受到敌意攻击。大多数关于图对抗攻击的研究都集中在图神经网络(GNN)上，而对HGNN上的对抗攻击的研究还很少。在本文中，我们试图缩小这一差距。针对非定向攻击，我们设计了一种新的HGNN攻击模型--MGHGA，该模型侧重于修改节点特征。我们考虑了HGNN的训练过程，并在超图建模之前使用代理模型来实现攻击。具体而言，MGHGA由特征选择和特征修改两部分组成。在特征选择模块中，我们使用动量梯度机制来选择攻击节点特征。在特征修改模块中，我们使用了两种特征生成方法(直接修改和符号梯度)，使得MGHGA可以在离散和连续的数据集上使用。我们在五个基准数据集上进行了大量的实验，以验证MGHGA在节点和视觉对象分类任务中的攻击性能。结果表明，与基线相比，MGHGA的性能平均提高了2%。



## **4. Deceptive Fairness Attacks on Graphs via Meta Learning**

基于元学习的图的欺骗性公平攻击 cs.LG

23 pages, 11 tables

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15653v1) [paper-pdf](http://arxiv.org/pdf/2310.15653v1)

**Authors**: Jian Kang, Yinglong Xia, Ross Maciejewski, Jiebo Luo, Hanghang Tong

**Abstract**: We study deceptive fairness attacks on graphs to answer the following question: How can we achieve poisoning attacks on a graph learning model to exacerbate the bias deceptively? We answer this question via a bi-level optimization problem and propose a meta learning-based framework named FATE. FATE is broadly applicable with respect to various fairness definitions and graph learning models, as well as arbitrary choices of manipulation operations. We further instantiate FATE to attack statistical parity and individual fairness on graph neural networks. We conduct extensive experimental evaluations on real-world datasets in the task of semi-supervised node classification. The experimental results demonstrate that FATE could amplify the bias of graph neural networks with or without fairness consideration while maintaining the utility on the downstream task. We hope this paper provides insights into the adversarial robustness of fair graph learning and can shed light on designing robust and fair graph learning in future studies.

摘要: 我们研究图上的欺骗性公平攻击，以回答以下问题：我们如何在图学习模型上实现中毒攻击，以欺骗性地加剧偏差？我们通过一个双层优化问题来回答这个问题，并提出了一个基于元学习的框架Fate。Fate广泛适用于各种公平性定义和图学习模型，以及任意选择的操作操作。我们进一步实例化命运来攻击图神经网络上的统计等价性和个体公平性。在半监督节点分类任务中，我们在真实数据集上进行了广泛的实验评估。实验结果表明，无论是否考虑公平性，Fate都能放大图神经网络的偏差，同时保持对下游任务的效用。我们希望本文能为公平图学习的对抗稳健性研究提供帮助，并在未来的研究中为设计稳健的公平图学习提供参考。



## **5. Facial Data Minimization: Shallow Model as Your Privacy Filter**

面部数据最小化：浅层模型作为您的隐私过滤器 cs.CR

14 pages, 11 figures

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15590v1) [paper-pdf](http://arxiv.org/pdf/2310.15590v1)

**Authors**: Yuwen Pu, Jiahao Chen, Jiayu Pan, Hao li, Diqun Yan, Xuhong Zhang, Shouling Ji

**Abstract**: Face recognition service has been used in many fields and brings much convenience to people. However, once the user's facial data is transmitted to a service provider, the user will lose control of his/her private data. In recent years, there exist various security and privacy issues due to the leakage of facial data. Although many privacy-preserving methods have been proposed, they usually fail when they are not accessible to adversaries' strategies or auxiliary data. Hence, in this paper, by fully considering two cases of uploading facial images and facial features, which are very typical in face recognition service systems, we proposed a data privacy minimization transformation (PMT) method. This method can process the original facial data based on the shallow model of authorized services to obtain the obfuscated data. The obfuscated data can not only maintain satisfactory performance on authorized models and restrict the performance on other unauthorized models but also prevent original privacy data from leaking by AI methods and human visual theft. Additionally, since a service provider may execute preprocessing operations on the received data, we also propose an enhanced perturbation method to improve the robustness of PMT. Besides, to authorize one facial image to multiple service models simultaneously, a multiple restriction mechanism is proposed to improve the scalability of PMT. Finally, we conduct extensive experiments and evaluate the effectiveness of the proposed PMT in defending against face reconstruction, data abuse, and face attribute estimation attacks. These experimental results demonstrate that PMT performs well in preventing facial data abuse and privacy leakage while maintaining face recognition accuracy.

摘要: 人脸识别服务已经在许多领域得到了应用，给人们带来了极大的便利。然而，一旦用户的面部数据被传输到服务提供商，用户将失去对他/她的私人数据的控制。近年来，由于人脸数据的泄露，存在着各种各样的安全和隐私问题。虽然已经提出了许多隐私保护方法，但当对手的策略或辅助数据无法访问时，这些方法通常会失败。因此，在本文中，通过充分考虑人脸识别服务系统中非常典型的两种上传人脸图像和人脸特征的情况，提出了一种数据隐私最小化转换(PMT)方法。该方法可以基于授权服务的浅模型对原始人脸数据进行处理，得到混淆后的数据。混淆后的数据不仅可以在授权模型上保持满意的性能，在其他非授权模型上也可以限制性能，还可以防止AI方法泄露原始隐私数据和人类视觉窃取。此外，由于服务提供商可以对接收到的数据执行预处理操作，我们还提出了一种增强的扰动方法来提高PMT的稳健性。此外，为了将一幅人脸图像同时授权给多个服务模型，提出了一种多约束机制来提高PMT的可扩展性。最后，我们进行了大量的实验，评估了提出的PMT在抵抗人脸重建、数据滥用和人脸属性估计攻击方面的有效性。这些实验结果表明，PMT在保持人脸识别准确率的同时，很好地防止了人脸数据的滥用和隐私泄露。



## **6. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

LLM自卫：通过自我检查，LLM知道自己被骗了 cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2308.07308v3) [paper-pdf](http://arxiv.org/pdf/2308.07308v3)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2.

摘要: 大型语言模型(LLM)对于高质量的文本生成很受欢迎，但可能会产生有害的内容，即使通过强化学习与人类的价值观保持一致。对抗性提示可以绕过它们的安全措施。我们提出了LLM自卫，这是一种通过让LLM筛选诱导响应来防御这些攻击的简单方法。我们的方法不需要任何微调、输入预处理或迭代输出生成。相反，我们将生成的内容合并到预定义的提示中，并使用LLM的另一个实例来分析文本并预测它是否有害。我们在GPT 3.5和Llama 2上测试了LLM自卫，这两种当前最著名的LLM针对各种类型的攻击，例如对提示的强制诱导肯定反应和提示工程攻击。值得注意的是，LLM自卫成功地使用GPT 3.5和Llama 2将攻击成功率降低到几乎为0。



## **7. Fast Propagation is Better: Accelerating Single-Step Adversarial Training via Sampling Subnetworks**

快速传播更好：通过抽样子网络加速单步对抗训练 cs.CV

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15444v1) [paper-pdf](http://arxiv.org/pdf/2310.15444v1)

**Authors**: Xiaojun Jia, Jianshu Li, Jindong Gu, Yang Bai, Xiaochun Cao

**Abstract**: Adversarial training has shown promise in building robust models against adversarial examples. A major drawback of adversarial training is the computational overhead introduced by the generation of adversarial examples. To overcome this limitation, adversarial training based on single-step attacks has been explored. Previous work improves the single-step adversarial training from different perspectives, e.g., sample initialization, loss regularization, and training strategy. Almost all of them treat the underlying model as a black box. In this work, we propose to exploit the interior building blocks of the model to improve efficiency. Specifically, we propose to dynamically sample lightweight subnetworks as a surrogate model during training. By doing this, both the forward and backward passes can be accelerated for efficient adversarial training. Besides, we provide theoretical analysis to show the model robustness can be improved by the single-step adversarial training with sampled subnetworks. Furthermore, we propose a novel sampling strategy where the sampling varies from layer to layer and from iteration to iteration. Compared with previous methods, our method not only reduces the training cost but also achieves better model robustness. Evaluations on a series of popular datasets demonstrate the effectiveness of the proposed FB-Better. Our code has been released at https://github.com/jiaxiaojunQAQ/FP-Better.

摘要: 对抗性训练在建立针对对抗性例子的健壮模型方面显示出了希望。对抗性训练的一个主要缺点是产生对抗性例子所带来的计算开销。为了克服这一局限性，基于单步攻击的对抗性训练被探索出来。以往的工作从样本初始化、损失正则化、训练策略等不同角度对单步对抗性训练进行了改进。几乎所有人都将基础模型视为一个黑匣子。在这项工作中，我们建议利用模型的内部构建块来提高效率。具体地说，我们提出了在训练过程中动态采样轻型子网络作为代理模型。通过这样做，向前和向后的传球都可以加快速度，以进行有效的对抗性训练。此外，我们还给出了理论分析，表明利用采样子网络进行单步对抗性训练可以提高模型的稳健性。此外，我们还提出了一种新的采样策略，即采样层次分明，迭代不同。与以前的方法相比，我们的方法不仅降低了训练成本，而且获得了更好的模型鲁棒性。在一系列流行数据集上的评估表明，所提出的FB-Better算法是有效的。我们的代码已在https://github.com/jiaxiaojunQAQ/FP-Better.发布



## **8. Unsupervised Federated Learning: A Federated Gradient EM Algorithm for Heterogeneous Mixture Models with Robustness against Adversarial Attacks**

无监督联合学习：一种适用于异质混合模型的联合梯度EM算法 stat.ML

43 pages, 1 figure

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15330v1) [paper-pdf](http://arxiv.org/pdf/2310.15330v1)

**Authors**: Ye Tian, Haolei Weng, Yang Feng

**Abstract**: While supervised federated learning approaches have enjoyed significant success, the domain of unsupervised federated learning remains relatively underexplored. In this paper, we introduce a novel federated gradient EM algorithm designed for the unsupervised learning of mixture models with heterogeneous mixture proportions across tasks. We begin with a comprehensive finite-sample theory that holds for general mixture models, then apply this general theory on Gaussian Mixture Models (GMMs) and Mixture of Regressions (MoRs) to characterize the explicit estimation error of model parameters and mixture proportions. Our proposed federated gradient EM algorithm demonstrates several key advantages: adaptability to unknown task similarity, resilience against adversarial attacks on a small fraction of data sources, protection of local data privacy, and computational and communication efficiency.

摘要: 虽然有监督的联合学习方法已经取得了很大的成功，但无监督的联合学习领域仍然相对较少被探索。本文提出了一种新的联邦梯度EM算法，用于混合模型跨任务混合比例的无监督学习。我们从一个适用于一般混合模型的综合有限样本理论开始，然后将这个一般理论应用于高斯混合模型(GMM)和混合回归模型(MORS)来刻画模型参数和混合比例的显式估计误差。我们提出的联合梯度EM算法具有以下几个关键优点：对未知任务相似性的适应性、对一小部分数据源的恶意攻击的恢复能力、对本地数据隐私的保护以及计算和通信效率。



## **9. GRASP: Accelerating Shortest Path Attacks via Graph Attention**

GRAPH：通过图注意力加速最短路径攻击 cs.LG

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.07980v2) [paper-pdf](http://arxiv.org/pdf/2310.07980v2)

**Authors**: Zohair Shafi, Benjamin A. Miller, Ayan Chatterjee, Tina Eliassi-Rad, Rajmonda S. Caceres

**Abstract**: Recent advances in machine learning (ML) have shown promise in aiding and accelerating classical combinatorial optimization algorithms. ML-based speed ups that aim to learn in an end to end manner (i.e., directly output the solution) tend to trade off run time with solution quality. Therefore, solutions that are able to accelerate existing solvers while maintaining their performance guarantees, are of great interest. We consider an APX-hard problem, where an adversary aims to attack shortest paths in a graph by removing the minimum number of edges. We propose the GRASP algorithm: Graph Attention Accelerated Shortest Path Attack, an ML aided optimization algorithm that achieves run times up to 10x faster, while maintaining the quality of solution generated. GRASP uses a graph attention network to identify a smaller subgraph containing the combinatorial solution, thus effectively reducing the input problem size. Additionally, we demonstrate how careful representation of the input graph, including node features that correlate well with the optimization task, can highlight important structure in the optimization solution.

摘要: 机器学习(ML)的最新进展在辅助和加速经典组合优化算法方面显示出良好的前景。基于ML的加速旨在以端到端的方式学习(即直接输出解决方案)，往往会在运行时间和解决方案质量之间进行权衡。因此，能够在保持现有求解器性能保证的同时加速现有求解器的解决方案是非常有意义的。我们考虑APX-Hard问题，其中对手的目标是通过删除最少的边数来攻击图中的最短路径。我们提出了GRASH算法：图注意加速最短路径攻击，这是一种ML辅助优化算法，在保持所生成解的质量的情况下，运行时间最多可以提高10倍。GRASH使用图注意网络来识别包含组合解的较小的子图，从而有效地减少了输入问题的规模。此外，我们还演示了如何仔细表示输入图，包括与优化任务关联良好的节点特征，以突出优化解决方案中的重要结构。



## **10. AutoDAN: Automatic and Interpretable Adversarial Attacks on Large Language Models**

AutoDAN：对大型语言模型的自动和可解释的对抗性攻击 cs.CR

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15140v1) [paper-pdf](http://arxiv.org/pdf/2310.15140v1)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent work suggests that patching LLMs against these attacks is possible: manual jailbreak attacks are human-readable but often limited and public, making them easy to block; adversarial attacks generate gibberish prompts that can be detected using perplexity-based filters. In this paper, we show that these solutions may be too optimistic. We propose an interpretable adversarial attack, \texttt{AutoDAN}, that combines the strengths of both types of attacks. It automatically generates attack prompts that bypass perplexity-based filters while maintaining a high attack success rate like manual jailbreak attacks. These prompts are interpretable and diverse, exhibiting strategies commonly used in manual jailbreak attacks, and transfer better than their non-readable counterparts when using limited training data or a single proxy model. We also customize \texttt{AutoDAN}'s objective to leak system prompts, another jailbreak application not addressed in the adversarial attack literature. %, demonstrating the versatility of the approach. We can also customize the objective of \texttt{AutoDAN} to leak system prompts, beyond the ability to elicit harmful content from the model, demonstrating the versatility of the approach. Our work provides a new way to red-team LLMs and to understand the mechanism of jailbreak attacks.

摘要: 大型语言模型(LLM)的安全对齐可能会受到手动越狱攻击和(自动)对抗性攻击的影响。最近的研究表明，修补LLM以抵御这些攻击是可能的：手动越狱攻击是人类可读的，但通常是有限的和公开的，使它们很容易被阻止；对抗性攻击生成胡言乱语的提示，可以使用基于困惑的过滤器检测到。在本文中，我们证明了这些解决方案可能过于乐观。我们提出了一种可解释的对抗性攻击，它结合了这两种攻击的优点。它自动生成攻击提示，绕过基于困惑的过滤器，同时保持较高的攻击成功率，如手动越狱攻击。这些提示是可解释的和多样化的，展示了手动越狱攻击中常用的策略，并且在使用有限的训练数据或单一代理模型时，传输效果比不可读的相应提示更好。我们还定制了S泄露系统提示的目标，这是另一个在对抗性攻击文献中没有涉及的越狱应用程序。%，展示了该方法的通用性。我们还可以定制泄露系统提示的目标，超越了从模型中引出有害内容的能力，展示了该方法的通用性。我们的工作为红队LMS和理解越狱攻击的机制提供了一种新的方法。



## **11. On the Detection of Image-Scaling Attacks in Machine Learning**

机器学习中图像缩放攻击的检测研究 cs.CR

Accepted at ACSAC'23

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15085v1) [paper-pdf](http://arxiv.org/pdf/2310.15085v1)

**Authors**: Erwin Quiring, Andreas Müller, Konrad Rieck

**Abstract**: Image scaling is an integral part of machine learning and computer vision systems. Unfortunately, this preprocessing step is vulnerable to so-called image-scaling attacks where an attacker makes unnoticeable changes to an image so that it becomes a new image after scaling. This opens up new ways for attackers to control the prediction or to improve poisoning and backdoor attacks. While effective techniques exist to prevent scaling attacks, their detection has not been rigorously studied yet. Consequently, it is currently not possible to reliably spot these attacks in practice.   This paper presents the first in-depth systematization and analysis of detection methods for image-scaling attacks. We identify two general detection paradigms and derive novel methods from them that are simple in design yet significantly outperform previous work. We demonstrate the efficacy of these methods in a comprehensive evaluation with all major learning platforms and scaling algorithms. First, we show that image-scaling attacks modifying the entire scaled image can be reliably detected even under an adaptive adversary. Second, we find that our methods provide strong detection performance even if only minor parts of the image are manipulated. As a result, we can introduce a novel protection layer against image-scaling attacks.

摘要: 图像缩放是机器学习和计算机视觉系统不可或缺的一部分。不幸的是，这一预处理步骤容易受到所谓的图像缩放攻击，即攻击者对图像进行不明显的更改，使其在缩放后成为新图像。这为攻击者控制预测或改进中毒和后门攻击开辟了新的途径。虽然已经有了有效的技术来防止伸缩攻击，但它们的检测还没有得到严格的研究。因此，目前还不可能在实践中可靠地发现这些攻击。本文首次对图像缩放攻击的检测方法进行了深入的整理和分析。我们确定了两个通用的检测范例，并从它们衍生出设计简单但显著优于以前工作的新方法。我们在与所有主要学习平台和缩放算法的综合评估中展示了这些方法的有效性。首先，我们证明了即使在自适应攻击下，修改了整个缩放图像的图像缩放攻击也可以被可靠地检测出来。其次，我们发现，我们的方法即使在图像的较小部分被篡改的情况下也能提供很强的检测性能。因此，我们可以引入一种新的保护层来抵御图像缩放攻击。



## **12. Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization**

对抗性不变正则化增强对抗性对比学习 cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2305.00374v2) [paper-pdf](http://arxiv.org/pdf/2305.00374v2)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL) is a technique that enhances standard contrastive learning (SCL) by incorporating adversarial data to learn a robust representation that can withstand adversarial attacks and common corruptions without requiring costly annotations. To improve transferability, the existing work introduced the standard invariant regularization (SIR) to impose style-independence property to SCL, which can exempt the impact of nuisance style factors in the standard representation. However, it is unclear how the style-independence property benefits ACL-learned robust representations. In this paper, we leverage the technique of causal reasoning to interpret the ACL and propose adversarial invariant regularization (AIR) to enforce independence from style factors. We regulate the ACL using both SIR and AIR to output the robust representation. Theoretically, we show that AIR implicitly encourages the representational distance between different views of natural data and their adversarial variants to be independent of style factors. Empirically, our experimental results show that invariant regularization significantly improves the performance of state-of-the-art ACL methods in terms of both standard generalization and robustness on downstream tasks. To the best of our knowledge, we are the first to apply causal reasoning to interpret ACL and develop AIR for enhancing ACL-learned robust representations. Our source code is at https://github.com/GodXuxilie/Enhancing_ACL_via_AIR.

摘要: 对抗性对比学习(ACL)是一种增强标准对比学习(SCL)的技术，它通过结合对抗性数据来学习健壮的表示，该表示可以抵抗对抗性攻击和常见的腐败，而不需要昂贵的注释。为了提高可转移性，现有的工作引入了标准不变正则化(SIR)，将风格无关性强加给SCL，从而可以免除标准表示中风格因素的影响。然而，尚不清楚样式独立属性如何使ACL学习的健壮表示受益。在本文中，我们利用因果推理技术来解释ACL，并提出了对抗不变正则化(AIR)来加强对风格因素的独立性。我们同时使用SIR和AIR来调节ACL，以输出稳健的表示。理论上，我们表明，AIR隐含地鼓励自然数据的不同观点与其敌对变体之间的表征距离独立于风格因素。实验结果表明，不变正则化在标准泛化和下游任务的稳健性方面都显著提高了最新的ACL方法的性能。据我们所知，我们是第一个应用因果推理来解释ACL并开发AIR来增强ACL学习的健壮表示的公司。我们的源代码在https://github.com/GodXuxilie/Enhancing_ACL_via_AIR.



## **13. Efficient Adversarial Contrastive Learning via Robustness-Aware Coreset Selection**

基于鲁棒性感知CoReset选择的高效对抗性对比学习 cs.LG

NeurIPS 2023 Spotlight

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2302.03857v4) [paper-pdf](http://arxiv.org/pdf/2302.03857v4)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL) does not require expensive data annotations but outputs a robust representation that withstands adversarial attacks and also generalizes to a wide range of downstream tasks. However, ACL needs tremendous running time to generate the adversarial variants of all training data, which limits its scalability to large datasets. To speed up ACL, this paper proposes a robustness-aware coreset selection (RCS) method. RCS does not require label information and searches for an informative subset that minimizes a representational divergence, which is the distance of the representation between natural data and their virtual adversarial variants. The vanilla solution of RCS via traversing all possible subsets is computationally prohibitive. Therefore, we theoretically transform RCS into a surrogate problem of submodular maximization, of which the greedy search is an efficient solution with an optimality guarantee for the original problem. Empirically, our comprehensive results corroborate that RCS can speed up ACL by a large margin without significantly hurting the robustness transferability. Notably, to the best of our knowledge, we are the first to conduct ACL efficiently on the large-scale ImageNet-1K dataset to obtain an effective robust representation via RCS. Our source code is at https://github.com/GodXuxilie/Efficient_ACL_via_RCS.

摘要: 对抗性对比学习(ACL)不需要昂贵的数据标注，但输出了一种稳健的表示，可以抵抗对抗性攻击，并适用于广泛的下游任务。然而，ACL需要大量的运行时间来生成所有训练数据的对抗性变体，这限制了其在大数据集上的可扩展性。为了提高访问控制列表的速度，提出了一种健壮性感知的核心重置选择(RCS)方法。RCS不需要标签信息，并且搜索最小化表示分歧的信息子集，表示分歧是自然数据和它们的虚拟对抗性变体之间的表示距离。通过遍历所有可能子集的RCS的香草解在计算上是令人望而却步的。因此，我们从理论上将RCS问题转化为子模最大化的代理问题，其中贪婪搜索是原问题的最优性保证的有效解。实验结果表明，RCS在不影响健壮性和可转移性的前提下，可以大幅度地提高ACL的速度。值得注意的是，据我们所知，我们是第一个在大规模ImageNet-1K数据集上高效地进行ACL的人，通过RCS获得了有效的健壮表示。我们的源代码在https://github.com/GodXuxilie/Efficient_ACL_via_RCS.



## **14. Kidnapping Deep Learning-based Multirotors using Optimized Flying Adversarial Patches**

优化飞行对抗性补丁绑架基于深度学习的多旋翼机器人 cs.RO

Accepted at MRS 2023, 7 pages, 5 figures. arXiv admin note:  substantial text overlap with arXiv:2305.12859

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2308.00344v2) [paper-pdf](http://arxiv.org/pdf/2308.00344v2)

**Authors**: Pia Hanfeld, Khaled Wahba, Marina M. -C. Höhne, Michael Bussmann, Wolfgang Hönig

**Abstract**: Autonomous flying robots, such as multirotors, often rely on deep learning models that make predictions based on a camera image, e.g. for pose estimation. These models can predict surprising results if applied to input images outside the training domain. This fault can be exploited by adversarial attacks, for example, by computing small images, so-called adversarial patches, that can be placed in the environment to manipulate the neural network's prediction. We introduce flying adversarial patches, where multiple images are mounted on at least one other flying robot and therefore can be placed anywhere in the field of view of a victim multirotor. By introducing the attacker robots, the system is extended to an adversarial multi-robot system. For an effective attack, we compare three methods that simultaneously optimize multiple adversarial patches and their position in the input image. We show that our methods scale well with the number of adversarial patches. Moreover, we demonstrate physical flights with two robots, where we employ a novel attack policy that uses the computed adversarial patches to kidnap a robot that was supposed to follow a human.

摘要: 自主飞行机器人（如多旋翼）通常依赖于深度学习模型，该模型基于相机图像进行预测，例如用于姿态估计。如果将这些模型应用于训练域之外的输入图像，则可以预测令人惊讶的结果。这种错误可以被对抗性攻击利用，例如，通过计算小图像，即所谓的对抗性补丁，可以将其放置在环境中以操纵神经网络的预测。我们介绍飞行对抗补丁，其中多个图像安装在至少一个其他飞行机器人，因此可以放置在任何地方的受害者多旋翼的视野。通过引入攻击机器人，系统扩展到一个对抗性的多机器人系统。对于有效的攻击，我们比较了三种同时优化多个对抗补丁及其在输入图像中的位置的方法。我们表明，我们的方法规模以及对抗补丁的数量。此外，我们演示了两个机器人的物理飞行，在那里我们采用了一种新的攻击策略，使用计算的对抗补丁来绑架一个应该跟随人类的机器人。



## **15. Beyond Hard Samples: Robust and Effective Grammatical Error Correction with Cycle Self-Augmenting**

超越硬样本：循环自增强的稳健有效的语法纠错 cs.CL

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.13321v2) [paper-pdf](http://arxiv.org/pdf/2310.13321v2)

**Authors**: Zecheng Tang, Kaifeng Qi, Juntao Li, Min Zhang

**Abstract**: Recent studies have revealed that grammatical error correction methods in the sequence-to-sequence paradigm are vulnerable to adversarial attack, and simply utilizing adversarial examples in the pre-training or post-training process can significantly enhance the robustness of GEC models to certain types of attack without suffering too much performance loss on clean data. In this paper, we further conduct a thorough robustness evaluation of cutting-edge GEC methods for four different types of adversarial attacks and propose a simple yet very effective Cycle Self-Augmenting (CSA) method accordingly. By leveraging the augmenting data from the GEC models themselves in the post-training process and introducing regularization data for cycle training, our proposed method can effectively improve the model robustness of well-trained GEC models with only a few more training epochs as an extra cost. More concretely, further training on the regularization data can prevent the GEC models from over-fitting on easy-to-learn samples and thus can improve the generalization capability and robustness towards unseen data (adversarial noise/samples). Meanwhile, the self-augmented data can provide more high-quality pseudo pairs to improve model performance on the original testing data. Experiments on four benchmark datasets and seven strong models indicate that our proposed training method can significantly enhance the robustness of four types of attacks without using purposely built adversarial examples in training. Evaluation results on clean data further confirm that our proposed CSA method significantly improves the performance of four baselines and yields nearly comparable results with other state-of-the-art models. Our code is available at https://github.com/ZetangForward/CSA-GEC.

摘要: 最近的研究表明，序列到序列范式中的语法纠错方法容易受到对抗性攻击，在训练前或训练后简单地使用对抗性例子可以显著增强GEC模型对某些类型攻击的鲁棒性，而不会在干净数据上造成太大的性能损失。在本文中，我们进一步对四种不同类型的对抗性攻击的前沿GEC方法进行了深入的健壮性评估，并相应地提出了一种简单但非常有效的循环自增强(CSA)方法。通过在训练后的过程中利用GEC模型本身增加的数据，并引入正则化数据进行循环训练，我们提出的方法可以有效地提高训练有素的GEC模型的模型稳健性，而只需要增加几个训练周期作为额外的代价。更具体地说，对正则化数据的进一步训练可以防止GEC模型对容易学习的样本进行过度拟合，从而提高对未知数据(对抗性噪声/样本)的泛化能力和鲁棒性。同时，自增强后的数据可以提供更多高质量的伪对，以提高模型在原始测试数据上的性能。在四个基准数据集和七个强模型上的实验表明，我们提出的训练方法可以显著提高四种类型攻击的稳健性，而不需要在训练中使用刻意构建的对抗性例子。对CLEAN数据的评估结果进一步证实，我们提出的CSA方法显著提高了四条基线的性能，并产生了与其他最先进模型几乎相当的结果。我们的代码可以在https://github.com/ZetangForward/CSA-GEC.上找到



## **16. Semantic-Aware Adversarial Training for Reliable Deep Hashing Retrieval**

面向可靠深度哈希检索的语义感知对抗性训练 cs.CV

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.14637v1) [paper-pdf](http://arxiv.org/pdf/2310.14637v1)

**Authors**: Xu Yuan, Zheng Zhang, Xunguang Wang, Lin Wu

**Abstract**: Deep hashing has been intensively studied and successfully applied in large-scale image retrieval systems due to its efficiency and effectiveness. Recent studies have recognized that the existence of adversarial examples poses a security threat to deep hashing models, that is, adversarial vulnerability. Notably, it is challenging to efficiently distill reliable semantic representatives for deep hashing to guide adversarial learning, and thereby it hinders the enhancement of adversarial robustness of deep hashing-based retrieval models. Moreover, current researches on adversarial training for deep hashing are hard to be formalized into a unified minimax structure. In this paper, we explore Semantic-Aware Adversarial Training (SAAT) for improving the adversarial robustness of deep hashing models. Specifically, we conceive a discriminative mainstay features learning (DMFL) scheme to construct semantic representatives for guiding adversarial learning in deep hashing. Particularly, our DMFL with the strict theoretical guarantee is adaptively optimized in a discriminative learning manner, where both discriminative and semantic properties are jointly considered. Moreover, adversarial examples are fabricated by maximizing the Hamming distance between the hash codes of adversarial samples and mainstay features, the efficacy of which is validated in the adversarial attack trials. Further, we, for the first time, formulate the formalized adversarial training of deep hashing into a unified minimax optimization under the guidance of the generated mainstay codes. Extensive experiments on benchmark datasets show superb attack performance against the state-of-the-art algorithms, meanwhile, the proposed adversarial training can effectively eliminate adversarial perturbations for trustworthy deep hashing-based retrieval. Our code is available at https://github.com/xandery-geek/SAAT.

摘要: 深度散列算法以其高效、高效的特点在大规模图像检索系统中得到了广泛的研究和成功的应用。最近的研究已经认识到，对抗性例子的存在对深度哈希模型构成了安全威胁，即对抗性漏洞。值得注意的是，有效地提取可靠的语义代表用于深度散列以指导对抗性学习是具有挑战性的，从而阻碍了基于深度散列的检索模型对抗性健壮性的增强。此外，目前针对深度散列的对抗性训练的研究很难被形式化成一个统一的极大极小结构。本文探讨了语义感知对抗训练(SAAT)来提高深度哈希模型的对抗健壮性。具体地说，我们设想了一种区分主干特征学习(DMFL)方案来构建语义表示，以指导深度哈希中的对抗性学习。特别是，我们的DMFL在严格的理论保证下，以区分学习的方式进行了自适应优化，同时考虑了区分属性和语义属性。此外，通过最大化对抗性样本的哈希码与主流特征之间的汉明距离来构造对抗性样本，并在对抗性攻击试验中验证了该方法的有效性。在生成的主干代码的指导下，首次将深度散列的形式化对抗性训练转化为统一的极大极小优化问题。在基准数据集上的大量实验表明，该算法对现有算法具有良好的攻击性能，同时，本文提出的对抗性训练能够有效地消除对抗性扰动，实现基于深度散列的可信检索。我们的代码可以在https://github.com/xandery-geek/SAAT.上找到



## **17. TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models**

TrojLLM：一种针对大型语言模型的黑盒木马提示攻击 cs.CR

Accepted by NeurIPS'23

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2306.06815v2) [paper-pdf](http://arxiv.org/pdf/2306.06815v2)

**Authors**: Jiaqi Xue, Mengxin Zheng, Ting Hua, Yilin Shen, Yepeng Liu, Ladislau Boloni, Qian Lou

**Abstract**: Large Language Models (LLMs) are progressively being utilized as machine learning services and interface tools for various applications. However, the security implications of LLMs, particularly in relation to adversarial and Trojan attacks, remain insufficiently examined. In this paper, we propose TrojLLM, an automatic and black-box framework to effectively generate universal and stealthy triggers. When these triggers are incorporated into the input data, the LLMs' outputs can be maliciously manipulated. Moreover, the framework also supports embedding Trojans within discrete prompts, enhancing the overall effectiveness and precision of the triggers' attacks. Specifically, we propose a trigger discovery algorithm for generating universal triggers for various inputs by querying victim LLM-based APIs using few-shot data samples. Furthermore, we introduce a novel progressive Trojan poisoning algorithm designed to generate poisoned prompts that retain efficacy and transferability across a diverse range of models. Our experiments and results demonstrate TrojLLM's capacity to effectively insert Trojans into text prompts in real-world black-box LLM APIs including GPT-3.5 and GPT-4, while maintaining exceptional performance on clean test sets. Our work sheds light on the potential security risks in current models and offers a potential defensive approach. The source code of TrojLLM is available at https://github.com/UCF-ML-Research/TrojLLM.

摘要: 大型语言模型(LLM)正逐渐被用作各种应用的机器学习服务和接口工具。然而，LLMS的安全影响，特别是与对抗性攻击和特洛伊木马攻击有关的影响，仍然没有得到充分的研究。在本文中，我们提出了一个自动黑盒框架TrojLLM，它可以有效地生成通用的、隐蔽的触发器。当这些触发器被合并到输入数据中时，LLMS的输出可能被恶意操纵。此外，该框架还支持在离散提示中嵌入特洛伊木马，增强了触发器攻击的整体有效性和精确度。具体地说，我们提出了一种触发器发现算法，通过使用少量数据样本查询受害者基于LLM的API来为各种输入生成通用触发器。此外，我们引入了一种新的渐进式特洛伊木马中毒算法，旨在生成中毒提示，从而在不同的模型中保持有效性和可转移性。我们的实验和结果表明，TrojLLM能够在包括GPT-3.5和GPT-4在内的真实黑盒LLMAPI中有效地将特洛伊木马程序插入到文本提示中，同时在干净的测试集上保持出色的性能。我们的工作揭示了当前模型中的潜在安全风险，并提供了一种潜在的防御方法。TrojLLm的源代码可在https://github.com/UCF-ML-Research/TrojLLM.上找到



## **18. ADoPT: LiDAR Spoofing Attack Detection Based on Point-Level Temporal Consistency**

采用：基于点级时间一致性的激光雷达欺骗攻击检测 cs.CV

BMVC 2023 (17 pages, 13 figures, and 1 table)

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.14504v1) [paper-pdf](http://arxiv.org/pdf/2310.14504v1)

**Authors**: Minkyoung Cho, Yulong Cao, Zixiang Zhou, Z. Morley Mao

**Abstract**: Deep neural networks (DNNs) are increasingly integrated into LiDAR (Light Detection and Ranging)-based perception systems for autonomous vehicles (AVs), requiring robust performance under adversarial conditions. We aim to address the challenge of LiDAR spoofing attacks, where attackers inject fake objects into LiDAR data and fool AVs to misinterpret their environment and make erroneous decisions. However, current defense algorithms predominantly depend on perception outputs (i.e., bounding boxes) thus face limitations in detecting attackers given the bounding boxes are generated by imperfect perception models processing limited points, acquired based on the ego vehicle's viewpoint. To overcome these limitations, we propose a novel framework, named ADoPT (Anomaly Detection based on Point-level Temporal consistency), which quantitatively measures temporal consistency across consecutive frames and identifies abnormal objects based on the coherency of point clusters. In our evaluation using the nuScenes dataset, our algorithm effectively counters various LiDAR spoofing attacks, achieving a low (< 10%) false positive ratio (FPR) and high (> 85%) true positive ratio (TPR), outperforming existing state-of-the-art defense methods, CARLO and 3D-TC2. Furthermore, our evaluation demonstrates the promising potential for accurate attack detection across various road environments.

摘要: 深度神经网络(DNN)越来越多地被集成到基于LiDAR(光检测和测距)的自主车辆(AV)感知系统中，要求在对抗条件下具有稳健的性能。我们的目标是应对LiDAR欺骗攻击的挑战，即攻击者向LiDAR数据中注入虚假对象，并愚弄AVs曲解其环境并做出错误的决定。然而，当前的防御算法主要依赖于感知输出(即包围盒)，因此在检测攻击者时面临限制，因为包围盒是由基于自我车辆的视角获得的处理有限点的不完美感知模型生成的。为了克服这些局限性，我们提出了一种新的框架，称为基于点级时间一致性的异常检测框架，该框架定量地测量连续帧的时间一致性，并根据点簇的一致性来识别异常对象。在我们使用nuScenes数据集进行的评估中，我们的算法有效地抵抗了各种LiDAR欺骗攻击，获得了低(<10%)的假正确率(FPR)和高(>85%)的真正确率(TPR)，性能优于现有的最先进的防御方法CALO和3D-TC2。此外，我们的评估显示了在各种道路环境中进行准确攻击检测的前景。



## **19. Probabilistic and Semantic Descriptions of Image Manifolds and Their Applications**

图像流形的概率和语义描述及其应用 cs.CV

26 pages, 17 figures, 1 table

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2307.02881v4) [paper-pdf](http://arxiv.org/pdf/2307.02881v4)

**Authors**: Peter Tu, Zhaoyuan Yang, Richard Hartley, Zhiwei Xu, Jing Zhang, Yiwei Fu, Dylan Campbell, Jaskirat Singh, Tianyu Wang

**Abstract**: This paper begins with a description of methods for estimating image probability density functions that reflects the observation that such data is usually constrained to lie in restricted regions of the high-dimensional image space-not every pattern of pixels is an image. It is common to say that images lie on a lower-dimensional manifold in the high-dimensional space. However, it is not the case that all points on the manifold have an equal probability of being images. Images are unevenly distributed on the manifold, and our task is to devise ways to model this distribution as a probability distribution. We therefore consider popular generative models. For our purposes, generative/probabilistic models should have the properties of 1) sample generation: the possibility to sample from this distribution with the modelled density function, and 2) probability computation: given a previously unseen sample from the dataset of interest, one should be able to compute its probability, at least up to a normalising constant. To this end, we investigate the use of methods such as normalising flow and diffusion models. We then show how semantic interpretations are used to describe points on the manifold. To achieve this, we consider an emergent language framework that uses variational encoders for a disentangled representation of points that reside on a given manifold. Trajectories between points on a manifold can then be described as evolving semantic descriptions. We also show that such probabilistic descriptions (bounded) can be used to improve semantic consistency by constructing defences against adversarial attacks. We evaluate our methods with improved semantic robustness and OoD detection capability, explainable and editable semantic interpolation, and improved classification accuracy under patch attacks. We also discuss the limitation in diffusion models.

摘要: 本文首先描述了用于估计图像概率密度函数的方法，该方法反映了这样的观察，即这种数据通常被限制在高维图像空间的受限区域-并不是每种像素模式都是图像。人们常说，图像位于高维空间中的低维流形上。然而，流形上的所有点成为图像的概率并不相等。图像在流形上是不均匀分布的，我们的任务是设计出将这种分布建模为概率分布的方法。因此，我们考虑流行的生成性模型。就我们的目的而言，生成/概率模型应该具有以下属性：1)样本生成：使用建模的密度函数从该分布中进行样本的可能性；以及2)概率计算：给定感兴趣的数据集中以前未见过的样本，应能够计算其概率，至少达到归一化常数。为此，我们研究了流和扩散模型等方法的使用。然后，我们展示了如何使用语义解释来描述流形上的点。为了实现这一点，我们考虑一种新的语言框架，它使用变分编码器来解开驻留在给定流形上的点的表示。然后，流形上的点之间的轨迹可以被描述为不断演变的语义描述。我们还表明，这种概率描述(有界)可以通过构建对对手攻击的防御来提高语义一致性。我们通过改进的语义健壮性和OOD检测能力、可解释和可编辑的语义内插以及在补丁攻击下改进的分类精度来评估我们的方法。我们还讨论了扩散模型的局限性。



## **20. Attacks Meet Interpretability (AmI) Evaluation and Findings**

攻击符合可解释性(AMI)评估和调查结果 cs.CR

Need to withdraw it. The current work needs to be changed at a large  extent which would take a longer time

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.08808v3) [paper-pdf](http://arxiv.org/pdf/2310.08808v3)

**Authors**: Qian Ma, Ziping Ye, Shagufta Mehnaz

**Abstract**: To investigate the effectiveness of the model explanation in detecting adversarial examples, we reproduce the results of two papers, Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples and Is AmI (Attacks Meet Interpretability) Robust to Adversarial Examples. And then conduct experiments and case studies to identify the limitations of both works. We find that Attacks Meet Interpretability(AmI) is highly dependent on the selection of hyperparameters. Therefore, with a different hyperparameter choice, AmI is still able to detect Nicholas Carlini's attack. Finally, we propose recommendations for future work on the evaluation of defense techniques such as AmI.

摘要: 为了考察模型解释在检测敌意实例方面的有效性，我们复制了两篇论文的结果：攻击满足解释性：对抗性样本的属性导向检测和AMI(攻击满足解释性)对对抗性实例的稳健性。然后进行实验和案例研究，找出两部作品的局限性。我们发现攻击满足可解释性(AMI)高度依赖于超参数的选择。因此，通过不同的超参数选择，阿米仍然能够检测到尼古拉斯·卡里尼的攻击。最后，我们对AMI等防御技术的未来评估工作提出了建议。



## **21. HoSNN: Adversarially-Robust Homeostatic Spiking Neural Networks with Adaptive Firing Thresholds**

HoSNN：具有自适应放电阈值的逆鲁棒自适应稳态尖峰神经网络 cs.NE

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2308.10373v2) [paper-pdf](http://arxiv.org/pdf/2308.10373v2)

**Authors**: Hejia Geng, Peng Li

**Abstract**: Spiking neural networks (SNNs) offer promise for efficient and powerful neurally inspired computation. Common to other types of neural networks, however, SNNs face the severe issue of vulnerability to adversarial attacks. We present the first study that draws inspiration from neural homeostasis to develop a bio-inspired solution that counters the susceptibilities of SNNs to adversarial onslaughts. At the heart of our approach is a novel threshold-adapting leaky integrate-and-fire (TA-LIF) neuron model, which we adopt to construct the proposed adversarially robust homeostatic SNN (HoSNN). Distinct from traditional LIF models, our TA-LIF model incorporates a self-stabilizing dynamic thresholding mechanism, curtailing adversarial noise propagation and safeguarding the robustness of HoSNNs in an unsupervised manner. Theoretical analysis is presented to shed light on the stability and convergence properties of the TA-LIF neurons, underscoring their superior dynamic robustness under input distributional shifts over traditional LIF neurons. Remarkably, without explicit adversarial training, our HoSNNs demonstrate inherent robustness on CIFAR-10, with accuracy improvements to 72.6% and 54.19% against FGSM and PGD attacks, up from 20.97% and 0.6%, respectively. Furthermore, with minimal FGSM adversarial training, our HoSNNs surpass previous models by 29.99% under FGSM and 47.83% under PGD attacks on CIFAR-10. Our findings offer a new perspective on harnessing biological principles for bolstering SNNs adversarial robustness and defense, paving the way to more resilient neuromorphic computing.

摘要: 尖峰神经网络(SNN)为高效和强大的神经启发计算提供了希望。然而，与其他类型的神经网络一样，SNN面临着易受对手攻击的严重问题。我们介绍了第一项从神经动态平衡中获得灵感的研究，以开发一种生物启发的解决方案，以对抗SNN对对手攻击的敏感性。我们方法的核心是一种新的阈值自适应泄漏积分与点火(TA-LIF)神经元模型，我们采用该模型来构造所提出的对抗性鲁棒自稳态SNN(HoSNN)。与传统的LIF模型不同，TA-LIF模型引入了一种自稳定的动态阈值机制，在无监督的情况下抑制了敌对噪声的传播，保护了HoSNN的健壮性。理论分析揭示了TA-LIF神经元的稳定性和收敛特性，强调了其在输入分布漂移下优于传统LIF神经元的动态鲁棒性。值得注意的是，在没有明确的对抗性训练的情况下，我们的HoSNN对CIFAR-10表现出固有的鲁棒性，对FGSM和PGD攻击的准确率分别从20.97%和0.6%提高到72.6%和54.19%。此外，在最少的FGSM对抗训练下，我们的HoSNN在FGSM下超过了以前的模型29.99%，在PGD攻击CIFAR-10下超过了47.83%。我们的发现为利用生物学原理加强SNN对抗的稳健性和防御提供了一个新的视角，为更具弹性的神经形态计算铺平了道路。



## **22. DANAA: Towards transferable attacks with double adversarial neuron attribution**

DANAA：具有双重对抗神经元属性的可转移攻击 cs.CV

Accepted by 19th International Conference on Advanced Data Mining and  Applications. (ADMA 2023)

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.10427v2) [paper-pdf](http://arxiv.org/pdf/2310.10427v2)

**Authors**: Zhibo Jin, Zhiyu Zhu, Xinyi Wang, Jiayu Zhang, Jun Shen, Huaming Chen

**Abstract**: While deep neural networks have excellent results in many fields, they are susceptible to interference from attacking samples resulting in erroneous judgments. Feature-level attacks are one of the effective attack types, which targets the learnt features in the hidden layers to improve its transferability across different models. Yet it is observed that the transferability has been largely impacted by the neuron importance estimation results. In this paper, a double adversarial neuron attribution attack method, termed `DANAA', is proposed to obtain more accurate feature importance estimation. In our method, the model outputs are attributed to the middle layer based on an adversarial non-linear path. The goal is to measure the weight of individual neurons and retain the features that are more important towards transferability. We have conducted extensive experiments on the benchmark datasets to demonstrate the state-of-the-art performance of our method. Our code is available at: https://github.com/Davidjinzb/DANAA

摘要: 虽然深度神经网络在许多领域都有很好的效果，但它们容易受到攻击样本的干扰，从而导致错误的判断。特征级攻击是一种有效的攻击类型，它针对隐含层中的学习特征，以提高其在不同模型上的可移植性。然而，观察到神经元重要性估计结果在很大程度上影响了神经网络的可转移性。为了获得更准确的特征重要性估计，本文提出了一种双重对抗神经元属性攻击方法--DANAA。在我们的方法中，模型输出被归因于基于对抗性非线性路径的中间层。其目标是测量单个神经元的重量，并保留对可转移性更重要的特征。我们在基准数据集上进行了广泛的实验，以证明我们的方法具有最先进的性能。我们的代码请访问：https://github.com/Davidjinzb/DANAA



## **23. Assessing the Influence of Different Types of Probing on Adversarial Decision-Making in a Deception Game**

在欺骗游戏中评估不同类型的探测对对抗性决策的影响 cs.CR

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.10662v2) [paper-pdf](http://arxiv.org/pdf/2310.10662v2)

**Authors**: Md Abu Sayed, Mohammad Ariful Islam Khan, Bryant A Allsup, Joshua Zamora, Palvi Aggarwal

**Abstract**: Deception, which includes leading cyber-attackers astray with false information, has shown to be an effective method of thwarting cyber-attacks. There has been little investigation of the effect of probing action costs on adversarial decision-making, despite earlier studies on deception in cybersecurity focusing primarily on variables like network size and the percentage of honeypots utilized in games. Understanding human decision-making when prompted with choices of various costs is essential in many areas such as in cyber security. In this paper, we will use a deception game (DG) to examine different costs of probing on adversarial decisions. To achieve this we utilized an IBLT model and a delayed feedback mechanism to mimic knowledge of human actions. Our results were taken from an even split of deception and no deception to compare each influence. It was concluded that probing was slightly taken less as the cost of probing increased. The proportion of attacks stayed relatively the same as the cost of probing increased. Although a constant cost led to a slight decrease in attacks. Overall, our results concluded that the different probing costs do not have an impact on the proportion of attacks whereas it had a slightly noticeable impact on the proportion of probing.

摘要: 欺骗，包括主要的网络攻击者误导虚假信息，已被证明是挫败网络攻击的有效方法。尽管早期关于网络安全中的欺骗的研究主要集中在网络规模和游戏中使用的蜜罐百分比等变量上，但关于探测行动成本对对抗性决策的影响的调查很少。在网络安全等许多领域，当被提示选择各种成本时，理解人类的决策是至关重要的。在本文中，我们将使用一个欺骗游戏(DG)来检查探测对手决策的不同成本。为了实现这一点，我们利用了IBLT模型和延迟反馈机制来模拟人类行为的知识。我们的结果取自欺骗性和非欺骗性的平分，以比较每种影响。结论是，随着探测成本的增加，探测的使用略有减少。随着探测成本的增加，攻击的比例保持相对不变。虽然不变的成本导致攻击略有减少。总体而言，我们的结果得出结论，不同的探测成本对攻击的比例没有影响，而对探测的比例有稍微明显的影响。



## **24. Language Model Unalignment: Parametric Red-Teaming to Expose Hidden Harms and Biases**

语言模型不一致：暴露隐藏的危害和偏见的参数红色团队 cs.CL

Under Review

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.14303v1) [paper-pdf](http://arxiv.org/pdf/2310.14303v1)

**Authors**: Rishabh Bhardwaj, Soujanya Poria

**Abstract**: Red-teaming has been a widely adopted way to evaluate the harmfulness of Large Language Models (LLMs). It aims to jailbreak a model's safety behavior to make it act as a helpful agent disregarding the harmfulness of the query. Existing methods are primarily based on input text-based red-teaming such as adversarial prompts, low-resource prompts, or contextualized prompts to condition the model in a way to bypass its safe behavior. Bypassing the guardrails uncovers hidden harmful information and biases in the model that are left untreated or newly introduced by its safety training. However, prompt-based attacks fail to provide such a diagnosis owing to their low attack success rate, and applicability to specific models. In this paper, we present a new perspective on LLM safety research i.e., parametric red-teaming through Unalignment. It simply (instruction) tunes the model parameters to break model guardrails that are not deeply rooted in the model's behavior. Unalignment using as few as 100 examples can significantly bypass commonly referred to as CHATGPT, to the point where it responds with an 88% success rate to harmful queries on two safety benchmark datasets. On open-source models such as VICUNA-7B and LLAMA-2-CHAT 7B AND 13B, it shows an attack success rate of more than 91%. On bias evaluations, Unalignment exposes inherent biases in safety-aligned models such as CHATGPT and LLAMA- 2-CHAT where the model's responses are strongly biased and opinionated 64% of the time.

摘要: 红团队已被广泛采用来评估大型语言模型的危害性。它的目的是让模特的安全行为越狱，使其成为一个有帮助的代理人，而不考虑询问的危害性。现有方法主要基于诸如对抗性提示、低资源提示或情境化提示的基于输入文本的红团队，以使模型以绕过其安全行为的方式调节。绕过护栏发现了模型中隐藏的有害信息和偏见，这些信息和偏见是未经处理的或安全培训新引入的。然而，基于提示的攻击无法提供这样的诊断，因为它们的攻击成功率低，并且适用于特定的模型。在这篇文章中，我们提出了一个新的视角来研究LLM安全，即通过非对齐的参数红组。它只是(指令)调整模型参数，以打破并不深深植根于模型行为中的模型护栏。只要使用100个例子，UnAlign就可以显著绕过通常所说的CHATGPT，以至于它对两个安全基准数据集上的有害查询的响应成功率为88%。在VIVUNA-7B和LLAMA-2-Chat 7B和13B等开源机型上，攻击成功率超过91%。在偏差评估方面，UnAlign暴露了安全对齐模型中的固有偏见，如CHATGPT和Llama-2-Chat，其中模型的反应在64%的时间内是强烈偏见和固执己见的。



## **25. CT-GAT: Cross-Task Generative Adversarial Attack based on Transferability**

CT-GAT：基于可转移性的跨任务生成性对抗攻击 cs.CL

Accepted to EMNLP 2023 main conference

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.14265v1) [paper-pdf](http://arxiv.org/pdf/2310.14265v1)

**Authors**: Minxuan Lv, Chengwei Dai, Kun Li, Wei Zhou, Songlin Hu

**Abstract**: Neural network models are vulnerable to adversarial examples, and adversarial transferability further increases the risk of adversarial attacks. Current methods based on transferability often rely on substitute models, which can be impractical and costly in real-world scenarios due to the unavailability of training data and the victim model's structural details. In this paper, we propose a novel approach that directly constructs adversarial examples by extracting transferable features across various tasks. Our key insight is that adversarial transferability can extend across different tasks. Specifically, we train a sequence-to-sequence generative model named CT-GAT using adversarial sample data collected from multiple tasks to acquire universal adversarial features and generate adversarial examples for different tasks. We conduct experiments on ten distinct datasets, and the results demonstrate that our method achieves superior attack performance with small cost.

摘要: 神经网络模型容易受到对抗性例子的影响，对抗性的可转移性进一步增加了对抗性攻击的风险。目前基于可转移性的方法往往依赖于替代模型，由于训练数据和受害者模型的结构细节的不可用，这在现实世界的场景中可能是不切实际和昂贵的。在本文中，我们提出了一种新的方法，该方法通过提取跨任务的可转移特征来直接构造对抗性实例。我们的关键洞察是，对抗性转移可以扩展到不同的任务。具体地说，我们使用从多个任务收集的对抗性样本数据来训练序列到序列生成模型CT-GAT，以获取通用的对抗性特征，并为不同的任务生成对抗性实例。我们在10个不同的数据集上进行了实验，结果表明，该方法以较小的代价获得了优越的攻击性能。



## **26. LoFT: Local Proxy Fine-tuning For Improving Transferability Of Adversarial Attacks Against Large Language Model**

LOFT：提高大型语言模型对抗性攻击可转移性的局部代理微调 cs.CL

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2310.04445v2) [paper-pdf](http://arxiv.org/pdf/2310.04445v2)

**Authors**: Muhammad Ahmed Shah, Roshan Sharma, Hira Dhamyal, Raphael Olivier, Ankit Shah, Joseph Konan, Dareen Alharthi, Hazim T Bukhari, Massa Baali, Soham Deshmukh, Michael Kuhlmann, Bhiksha Raj, Rita Singh

**Abstract**: It has been shown that Large Language Model (LLM) alignments can be circumvented by appending specially crafted attack suffixes with harmful queries to elicit harmful responses. To conduct attacks against private target models whose characterization is unknown, public models can be used as proxies to fashion the attack, with successful attacks being transferred from public proxies to private target models. The success rate of attack depends on how closely the proxy model approximates the private model. We hypothesize that for attacks to be transferrable, it is sufficient if the proxy can approximate the target model in the neighborhood of the harmful query. Therefore, in this paper, we propose \emph{Local Fine-Tuning (LoFT)}, \textit{i.e.}, fine-tuning proxy models on similar queries that lie in the lexico-semantic neighborhood of harmful queries to decrease the divergence between the proxy and target models. First, we demonstrate three approaches to prompt private target models to obtain similar queries given harmful queries. Next, we obtain data for local fine-tuning by eliciting responses from target models for the generated similar queries. Then, we optimize attack suffixes to generate attack prompts and evaluate the impact of our local fine-tuning on the attack's success rate. Experiments show that local fine-tuning of proxy models improves attack transferability and increases attack success rate by $39\%$, $7\%$, and $0.5\%$ (absolute) on target models ChatGPT, GPT-4, and Claude respectively.

摘要: 已有研究表明，通过在巧尽心思构建的攻击后缀上附加有害查询来引发有害响应，可以绕过大型语言模型(LLM)对齐。为了对特征未知的私有目标模型进行攻击，可以使用公共模型作为代理来进行攻击，成功的攻击将从公共代理转移到私有目标模型。攻击的成功率取决于代理模型与私有模型的接近程度。我们假设，对于可转移的攻击，只要代理能够逼近有害查询附近的目标模型就足够了。因此，在本文中，我们提出了对位于有害查询的词典-语义邻域中的相似查询的代理模型进行微调，以减少代理模型和目标模型之间的差异。首先，我们演示了三种方法来提示私人目标模型在给定有害查询的情况下获得类似的查询。接下来，我们通过从目标模型获取对生成的类似查询的响应来获得用于本地微调的数据。然后，我们优化攻击后缀来生成攻击提示，并评估我们的局部微调对攻击成功率的影响。实验表明，代理模型的局部微调提高了攻击的可转移性，使目标模型ChatGPT、GPT-4和Claude的攻击成功率分别提高了39美元、7美元和0.5美元(绝对)。



## **27. Toward Stronger Textual Attack Detectors**

走向更强大的文本攻击检测器 cs.CL

Findings EMNLP 2023

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2310.14001v1) [paper-pdf](http://arxiv.org/pdf/2310.14001v1)

**Authors**: Pierre Colombo, Marine Picot, Nathan Noiry, Guillaume Staerman, Pablo Piantanida

**Abstract**: The landscape of available textual adversarial attacks keeps growing, posing severe threats and raising concerns regarding the deep NLP system's integrity. However, the crucial problem of defending against malicious attacks has only drawn the attention of the NLP community. The latter is nonetheless instrumental in developing robust and trustworthy systems. This paper makes two important contributions in this line of search: (i) we introduce LAROUSSE, a new framework to detect textual adversarial attacks and (ii) we introduce STAKEOUT, a new benchmark composed of nine popular attack methods, three datasets, and two pre-trained models. LAROUSSE is ready-to-use in production as it is unsupervised, hyperparameter-free, and non-differentiable, protecting it against gradient-based methods. Our new benchmark STAKEOUT allows for a robust evaluation framework: we conduct extensive numerical experiments which demonstrate that LAROUSSE outperforms previous methods, and which allows to identify interesting factors of detection rate variations.

摘要: 可用的文本对抗性攻击的情况不断增长，构成了严重的威胁，并引发了对深度NLP系统完整性的担忧。然而，防御恶意攻击的关键问题只引起了NLP社区的关注。尽管如此，后者在开发强大和值得信赖的系统方面仍然起到了重要作用。本文在这方面做了两个重要的贡献：(I)我们介绍了Larousse，一个新的文本攻击检测框架；(Ii)我们介绍了一个新的基准，它由九种流行的攻击方法，三个数据集和两个预先训练的模型组成。Larousse在生产中随时可用，因为它是无监督、无超参数和不可区分的，可以保护它免受基于梯度的方法的影响。我们的新基准监视允许一个强大的评估框架：我们进行了广泛的数值实验，证明Larousse的性能优于以前的方法，并允许识别检测率变化的有趣因素。



## **28. Adversarial Image Generation by Spatial Transformation in Perceptual Colorspaces**

感知色彩空间中基于空间变换的对抗性图像生成 cs.CV

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2310.13950v1) [paper-pdf](http://arxiv.org/pdf/2310.13950v1)

**Authors**: Ayberk Aydin, Alptekin Temizel

**Abstract**: Deep neural networks are known to be vulnerable to adversarial perturbations. The amount of these perturbations are generally quantified using $L_p$ metrics, such as $L_0$, $L_2$ and $L_\infty$. However, even when the measured perturbations are small, they tend to be noticeable by human observers since $L_p$ distance metrics are not representative of human perception. On the other hand, humans are less sensitive to changes in colorspace. In addition, pixel shifts in a constrained neighborhood are hard to notice. Motivated by these observations, we propose a method that creates adversarial examples by applying spatial transformations, which creates adversarial examples by changing the pixel locations independently to chrominance channels of perceptual colorspaces such as $YC_{b}C_{r}$ and $CIELAB$, instead of making an additive perturbation or manipulating pixel values directly. In a targeted white-box attack setting, the proposed method is able to obtain competitive fooling rates with very high confidence. The experimental evaluations show that the proposed method has favorable results in terms of approximate perceptual distance between benign and adversarially generated images. The source code is publicly available at https://github.com/ayberkydn/stadv-torch

摘要: 众所周知，深度神经网络容易受到对抗性扰动的影响。这些扰动的量通常使用$L_p$度量来量化，例如$L_0$、$L_2$和$L_\inty$。然而，即使测量到的扰动很小，它们往往也会被人类观察者注意到，因为$L_p$距离度量不能代表人类的感知。另一方面，人类对色彩空间的变化不那么敏感。此外，受限制的邻域中的像素移动很难注意到。基于这些观察结果，我们提出了一种通过应用空间变换来创建对抗性示例的方法，该方法通过独立于感知色彩空间的色度通道改变像素位置来创建对抗性示例，例如$YCb}C{r}$和$CIELAB$，而不是直接进行加性扰动或操作像素值。在有针对性的白盒攻击环境下，该方法能够以很高的置信度获得具有竞争力的欺骗率。实验结果表明，该方法在良性图像和恶意图像之间的近似感知距离方面具有较好的效果。源代码可在https://github.com/ayberkydn/stadv-torch上公开获得



## **29. Adversarial Robustness Unhardening via Backdoor Attacks in Federated Learning**

联合学习中通过后门攻击的对抗性健壮性去坚固性 cs.LG

8 pages, 6 main pages of text, 4 figures, 2 tables. Made for a  Neurips workshop on backdoor attacks

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2310.11594v2) [paper-pdf](http://arxiv.org/pdf/2310.11594v2)

**Authors**: Taejin Kim, Jiarui Li, Shubhranshu Singh, Nikhil Madaan, Carlee Joe-Wong

**Abstract**: In today's data-driven landscape, the delicate equilibrium between safeguarding user privacy and unleashing data potential stands as a paramount concern. Federated learning, which enables collaborative model training without necessitating data sharing, has emerged as a privacy-centric solution. This decentralized approach brings forth security challenges, notably poisoning and backdoor attacks where malicious entities inject corrupted data. Our research, initially spurred by test-time evasion attacks, investigates the intersection of adversarial training and backdoor attacks within federated learning, introducing Adversarial Robustness Unhardening (ARU). ARU is employed by a subset of adversaries to intentionally undermine model robustness during decentralized training, rendering models susceptible to a broader range of evasion attacks. We present extensive empirical experiments evaluating ARU's impact on adversarial training and existing robust aggregation defenses against poisoning and backdoor attacks. Our findings inform strategies for enhancing ARU to counter current defensive measures and highlight the limitations of existing defenses, offering insights into bolstering defenses against ARU.

摘要: 在当今数据驱动的格局中，保护用户隐私和释放数据潜力之间的微妙平衡是一个最重要的问题。联合学习是一种以隐私为中心的解决方案，它能够在不需要共享数据的情况下进行协作模型培训。这种分散的方法带来了安全挑战，尤其是毒化和后门攻击，即恶意实体注入被破坏的数据。我们的研究最初受到测试时间逃避攻击的启发，研究了联邦学习中对抗性训练和后门攻击的交集，引入了对抗性健壮性不硬化(ARU)。ARU被一部分攻击者利用来在分散训练期间故意破坏模型的健壮性，使模型容易受到更大范围的逃避攻击。我们提供了广泛的经验实验，评估ARU对对手训练的影响，以及现有针对中毒和后门攻击的健壮聚合防御。我们的发现为增强ARU以对抗当前防御措施的战略提供了参考，并突出了现有防御的局限性，为加强对ARU的防御提供了见解。



## **30. Characterizing Internal Evasion Attacks in Federated Learning**

联合学习中内部逃避攻击的特征 cs.LG

16 pages, 8 figures (14 images if counting sub-figures separately),  Camera ready version for AISTATS 2023, longer version of paper submitted to  CrossFL 2022 poster workshop, code available at  (https://github.com/tj-kim/pFedDef_v1)

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2209.08412v3) [paper-pdf](http://arxiv.org/pdf/2209.08412v3)

**Authors**: Taejin Kim, Shubhranshu Singh, Nikhil Madaan, Carlee Joe-Wong

**Abstract**: Federated learning allows for clients in a distributed system to jointly train a machine learning model. However, clients' models are vulnerable to attacks during the training and testing phases. In this paper, we address the issue of adversarial clients performing "internal evasion attacks": crafting evasion attacks at test time to deceive other clients. For example, adversaries may aim to deceive spam filters and recommendation systems trained with federated learning for monetary gain. The adversarial clients have extensive information about the victim model in a federated learning setting, as weight information is shared amongst clients. We are the first to characterize the transferability of such internal evasion attacks for different learning methods and analyze the trade-off between model accuracy and robustness depending on the degree of similarities in client data. We show that adversarial training defenses in the federated learning setting only display limited improvements against internal attacks. However, combining adversarial training with personalized federated learning frameworks increases relative internal attack robustness by 60% compared to federated adversarial training and performs well under limited system resources.

摘要: 联合学习允许分布式系统中的客户端联合训练机器学习模型。然而，客户的模型在培训和测试阶段很容易受到攻击。在本文中，我们讨论了敌意客户执行“内部逃避攻击”的问题：在测试时精心设计逃避攻击以欺骗其他客户。例如，对手的目标可能是欺骗经过联合学习培训的垃圾邮件过滤器和推荐系统，以换取金钱利益。当权重信息在客户端之间共享时，敌意客户端在联合学习环境中具有关于受害者模型的大量信息。我们首次针对不同的学习方法刻画了这种内部规避攻击的可转移性，并根据客户数据的相似程度分析了模型精度和稳健性之间的权衡。我们表明，在联合学习环境中，对抗性训练防御对内部攻击的改善有限。然而，将对抗性训练与个性化的联合学习框架相结合，与联合对抗性训练相比，相对内部攻击健壮性提高了60%，并且在有限的系统资源下表现良好。



## **31. The Hidden Adversarial Vulnerabilities of Medical Federated Learning**

医学联合学习中隐藏的对抗性弱点 cs.LG

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2310.13893v1) [paper-pdf](http://arxiv.org/pdf/2310.13893v1)

**Authors**: Erfan Darzi, Florian Dubost, Nanna. M. Sijtsema, P. M. A van Ooijen

**Abstract**: In this paper, we delve into the susceptibility of federated medical image analysis systems to adversarial attacks. Our analysis uncovers a novel exploitation avenue: using gradient information from prior global model updates, adversaries can enhance the efficiency and transferability of their attacks. Specifically, we demonstrate that single-step attacks (e.g. FGSM), when aptly initialized, can outperform the efficiency of their iterative counterparts but with reduced computational demand. Our findings underscore the need to revisit our understanding of AI security in federated healthcare settings.

摘要: 在本文中，我们深入研究了联合医学图像分析系统对敌意攻击的敏感性。我们的分析揭示了一种新的攻击途径：利用先前全局模型更新的梯度信息，攻击者可以提高其攻击的效率和可转移性。具体地说，我们证明了单步攻击(例如FGSM)，当适当地初始化时，可以在计算需求减少的情况下比它们的迭代对应攻击的效率更高。我们的发现强调了重新审视我们对联邦医疗保健环境中人工智能安全的理解的必要性。



## **32. Specify Robust Causal Representation from Mixed Observations**

从混合观测中指定稳健的因果表示 cs.LG

arXiv admin note: substantial text overlap with arXiv:2202.08388

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2310.13892v1) [paper-pdf](http://arxiv.org/pdf/2310.13892v1)

**Authors**: Mengyue Yang, Xinyu Cai, Furui Liu, Weinan Zhang, Jun Wang

**Abstract**: Learning representations purely from observations concerns the problem of learning a low-dimensional, compact representation which is beneficial to prediction models. Under the hypothesis that the intrinsic latent factors follow some casual generative models, we argue that by learning a causal representation, which is the minimal sufficient causes of the whole system, we can improve the robustness and generalization performance of machine learning models. In this paper, we develop a learning method to learn such representation from observational data by regularizing the learning procedure with mutual information measures, according to the hypothetical factored causal graph. We theoretically and empirically show that the models trained with the learned causal representations are more robust under adversarial attacks and distribution shifts compared with baselines. The supplementary materials are available at https://github.com/ymy $4323460 / \mathrm{CaRI} /$.

摘要: 纯粹从观测中学习表示涉及到学习低维的、紧凑的表示的问题，这对预测模型是有益的。在假设内在潜在因素遵循一些随机生成模型的假设下，我们认为通过学习一个因果表示可以提高机器学习模型的稳健性和泛化性能。在本文中，我们发展了一种学习方法来从观测数据中学习这种表示，方法是根据假设的因果图，用互信息度量来规范学习过程。我们的理论和经验表明，与基线相比，使用学习的因果表示训练的模型在对抗攻击和分布偏移的情况下具有更好的鲁棒性。补充材料可在https://github.com/ymy$4323460/\mathm{cari}/$上查阅。



## **33. Adversarial Attacks on Fairness of Graph Neural Networks**

图神经网络公平性的对抗性攻击 cs.LG

32 pages, 5 figures

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13822v1) [paper-pdf](http://arxiv.org/pdf/2310.13822v1)

**Authors**: Binchi Zhang, Yushun Dong, Chen Chen, Yada Zhu, Minnan Luo, Jundong Li

**Abstract**: Fairness-aware graph neural networks (GNNs) have gained a surge of attention as they can reduce the bias of predictions on any demographic group (e.g., female) in graph-based applications. Although these methods greatly improve the algorithmic fairness of GNNs, the fairness can be easily corrupted by carefully designed adversarial attacks. In this paper, we investigate the problem of adversarial attacks on fairness of GNNs and propose G-FairAttack, a general framework for attacking various types of fairness-aware GNNs in terms of fairness with an unnoticeable effect on prediction utility. In addition, we propose a fast computation technique to reduce the time complexity of G-FairAttack. The experimental study demonstrates that G-FairAttack successfully corrupts the fairness of different types of GNNs while keeping the attack unnoticeable. Our study on fairness attacks sheds light on potential vulnerabilities in fairness-aware GNNs and guides further research on the robustness of GNNs in terms of fairness. The open-source code is available at https://github.com/zhangbinchi/G-FairAttack.

摘要: 在基于图的应用中，公平感知图神经网络(GNN)可以减少对任何人口统计群体(例如，女性)的预测偏差，因此受到了广泛的关注。虽然这些方法极大地提高了GNN算法的公平性，但这种公平性很容易被精心设计的对抗性攻击所破坏。本文研究了对GNN公平性的敌意攻击问题，提出了G-FairAttack框架，该框架从公平性的角度攻击各种类型的公平感知GNN，并且对预测效用没有明显的影响。此外，我们还提出了一种快速计算技术来降低G-FairAttack的时间复杂度。实验研究表明，G-FairAttack在保持攻击不可察觉的同时，成功地破坏了不同类型GNN的公平性。我们对公平攻击的研究揭示了公平感知网络中潜在的漏洞，并指导了进一步研究公平感知网络的健壮性。开放源码可在https://github.com/zhangbinchi/G-FairAttack.上获得



## **34. Data-Free Knowledge Distillation Using Adversarially Perturbed OpenGL Shader Images**

基于逆扰动OpenGL着色器图像的无数据知识提取 cs.CV

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13782v1) [paper-pdf](http://arxiv.org/pdf/2310.13782v1)

**Authors**: Logan Frank, Jim Davis

**Abstract**: Knowledge distillation (KD) has been a popular and effective method for model compression. One important assumption of KD is that the original training dataset is always available. However, this is not always the case due to privacy concerns and more. In recent years, "data-free" KD has emerged as a growing research topic which focuses on the scenario of performing KD when no data is provided. Many methods rely on a generator network to synthesize examples for distillation (which can be difficult to train) and can frequently produce images that are visually similar to the original dataset, which raises questions surrounding whether privacy is completely preserved. In this work, we propose a new approach to data-free KD that utilizes unnatural OpenGL images, combined with large amounts of data augmentation and adversarial attacks, to train a student network. We demonstrate that our approach achieves state-of-the-art results for a variety of datasets/networks and is more stable than existing generator-based data-free KD methods. Source code will be available in the future.

摘要: 知识蒸馏(KD)是一种流行而有效的模型压缩方法。KD的一个重要假设是原始训练数据集始终可用。然而，出于隐私等方面的考虑，情况并不总是如此。近年来，无数据KD逐渐成为一个研究热点，主要研究在没有数据提供的情况下执行KD的场景。许多方法依靠生成器网络来合成用于蒸馏的示例(这可能很难训练)，并且经常可以产生与原始数据集在视觉上相似的图像，这引发了关于隐私是否完全得到保护的问题。在这项工作中，我们提出了一种新的无数据KD方法，该方法利用非自然的OpenGL图像，结合大量的数据增强和敌意攻击来训练学生网络。我们证明了我们的方法在各种数据集/网络上获得了最先进的结果，并且比现有的基于生成器的无数据KD方法更稳定。源代码将在未来提供。



## **35. FLTracer: Accurate Poisoning Attack Provenance in Federated Learning**

FLTracer：联邦学习中准确的中毒攻击来源 cs.CR

18 pages, 27 figures

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13424v1) [paper-pdf](http://arxiv.org/pdf/2310.13424v1)

**Authors**: Xinyu Zhang, Qingyu Liu, Zhongjie Ba, Yuan Hong, Tianhang Zheng, Feng Lin, Li Lu, Kui Ren

**Abstract**: Federated Learning (FL) is a promising distributed learning approach that enables multiple clients to collaboratively train a shared global model. However, recent studies show that FL is vulnerable to various poisoning attacks, which can degrade the performance of global models or introduce backdoors into them. In this paper, we first conduct a comprehensive study on prior FL attacks and detection methods. The results show that all existing detection methods are only effective against limited and specific attacks. Most detection methods suffer from high false positives, which lead to significant performance degradation, especially in not independent and identically distributed (non-IID) settings. To address these issues, we propose FLTracer, the first FL attack provenance framework to accurately detect various attacks and trace the attack time, objective, type, and poisoned location of updates. Different from existing methodologies that rely solely on cross-client anomaly detection, we propose a Kalman filter-based cross-round detection to identify adversaries by seeking the behavior changes before and after the attack. Thus, this makes it resilient to data heterogeneity and is effective even in non-IID settings. To further improve the accuracy of our detection method, we employ four novel features and capture their anomalies with the joint decisions. Extensive evaluations show that FLTracer achieves an average true positive rate of over $96.88\%$ at an average false positive rate of less than $2.67\%$, significantly outperforming SOTA detection methods. \footnote{Code is available at \url{https://github.com/Eyr3/FLTracer}.}

摘要: 联合学习(FL)是一种很有前途的分布式学习方法，它使多个客户能够协作地训练一个共享的全局模型。然而，最近的研究表明，FL很容易受到各种中毒攻击，这些攻击可能会降低全局模型的性能，或者在其中引入后门。本文首先对现有的FL攻击和检测方法进行了全面的研究。结果表明，现有的所有检测方法只对有限和特定的攻击有效。大多数检测方法都存在较高的误报，这会导致性能显著下降，特别是在不独立且相同分布(非IID)的环境中。为了解决这些问题，我们提出了第一个FL攻击来源框架FLTracer，它可以准确地检测各种攻击，并跟踪更新的攻击时间、目标、类型和中毒位置。不同于现有的单一依赖于跨客户端异常检测的方法，本文提出了一种基于卡尔曼滤波的跨轮检测方法，通过寻找攻击前后的行为变化来识别对手。因此，这使得它对数据异构性具有弹性，即使在非IID设置中也是有效的。为了进一步提高检测方法的准确性，我们采用了四个新的特征，并使用联合判决来捕获它们的异常。广泛的测试表明，该算法的平均真阳性率大于96 88美元，平均假阳性率低于2 67美元，显著优于SOTA检测方法。\脚注{代码位于\url{https://github.com/Eyr3/FLTracer}.}



## **36. Vulnerabilities in Video Quality Assessment Models: The Challenge of Adversarial Attacks**

视频质量评估模型中的漏洞：对抗性攻击的挑战 cs.CV

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2309.13609v3) [paper-pdf](http://arxiv.org/pdf/2309.13609v3)

**Authors**: Ao-Xiang Zhang, Yu Ran, Weixuan Tang, Yuan-Gen Wang

**Abstract**: No-Reference Video Quality Assessment (NR-VQA) plays an essential role in improving the viewing experience of end-users. Driven by deep learning, recent NR-VQA models based on Convolutional Neural Networks (CNNs) and Transformers have achieved outstanding performance. To build a reliable and practical assessment system, it is of great necessity to evaluate their robustness. However, such issue has received little attention in the academic community. In this paper, we make the first attempt to evaluate the robustness of NR-VQA models against adversarial attacks, and propose a patch-based random search method for black-box attack. Specifically, considering both the attack effect on quality score and the visual quality of adversarial video, the attack problem is formulated as misleading the estimated quality score under the constraint of just-noticeable difference (JND). Built upon such formulation, a novel loss function called Score-Reversed Boundary Loss is designed to push the adversarial video's estimated quality score far away from its ground-truth score towards a specific boundary, and the JND constraint is modeled as a strict $L_2$ and $L_\infty$ norm restriction. By this means, both white-box and black-box attacks can be launched in an effective and imperceptible manner. The source code is available at https://github.com/GZHU-DVL/AttackVQA.

摘要: 无参考视频质量评估(NR-VQA)对于改善终端用户的观看体验起着至关重要的作用。在深度学习的推动下，最近基于卷积神经网络(CNN)和变压器的NR-VQA模型取得了优异的性能。为了建立一个可靠、实用的评估体系，对它们的稳健性进行评估是非常必要的。然而，这一问题在学术界却鲜有人关注。本文首次尝试评估了NR-VQA模型对对手攻击的稳健性，并提出了一种基于补丁的黑盒攻击随机搜索方法。具体地说，综合考虑攻击对视频质量分数的影响和对抗性视频的视觉质量，在JND约束下，将攻击问题描述为误导估计质量分数。在此基础上，设计了一种新的损失函数--分数反向边界损失函数，将对抗性视频的估计质量分数远离其真实分数推向特定的边界，并将JND约束建模为严格的$L_2$和$L_\inty$范数限制。通过这种手段，白盒攻击和黑盒攻击都可以有效地、隐蔽地发动。源代码可在https://github.com/GZHU-DVL/AttackVQA.上找到



## **37. Verifiable Learning for Robust Tree Ensembles**

用于稳健树集成的可验证学习 cs.LG

19 pages, 5 figures; full version of the revised paper accepted at  ACM CCS 2023 with corrected proofs of Lemma A.6 and Lemma A.7

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2305.03626v3) [paper-pdf](http://arxiv.org/pdf/2305.03626v3)

**Authors**: Stefano Calzavara, Lorenzo Cazzaro, Giulio Ermanno Pibiri, Nicola Prezza

**Abstract**: Verifying the robustness of machine learning models against evasion attacks at test time is an important research problem. Unfortunately, prior work established that this problem is NP-hard for decision tree ensembles, hence bound to be intractable for specific inputs. In this paper, we identify a restricted class of decision tree ensembles, called large-spread ensembles, which admit a security verification algorithm running in polynomial time. We then propose a new approach called verifiable learning, which advocates the training of such restricted model classes which are amenable for efficient verification. We show the benefits of this idea by designing a new training algorithm that automatically learns a large-spread decision tree ensemble from labelled data, thus enabling its security verification in polynomial time. Experimental results on public datasets confirm that large-spread ensembles trained using our algorithm can be verified in a matter of seconds, using standard commercial hardware. Moreover, large-spread ensembles are more robust than traditional ensembles against evasion attacks, at the cost of an acceptable loss of accuracy in the non-adversarial setting.

摘要: 验证机器学习模型在测试时对逃避攻击的稳健性是一个重要的研究问题。不幸的是，以前的工作确定了这个问题对于决策树集成来说是NP-Hard的，因此对于特定的输入必然是棘手的。在本文中，我们识别了一类受限的决策树集成，称为大分布集成，它允许安全验证算法在多项式时间内运行。然后，我们提出了一种新的方法，称为可验证学习，它主张训练这样的受限模型类，这些模型类适合于有效的验证。我们通过设计一种新的训练算法，从标记数据中自动学习大规模决策树集成，从而在多项式时间内实现其安全性验证，从而展示了这种思想的好处。在公共数据集上的实验结果证实，使用我们的算法训练的大范围集成可以在几秒钟内使用标准的商业硬件进行验证。此外，大范围的合奏比传统的合奏更能抵抗躲避攻击，代价是在非对抗性环境中损失可接受的准确性。



## **38. An LLM can Fool Itself: A Prompt-Based Adversarial Attack**

LLM可以自欺欺人：基于提示的对抗性攻击 cs.CR

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13345v1) [paper-pdf](http://arxiv.org/pdf/2310.13345v1)

**Authors**: Xilie Xu, Keyi Kong, Ning Liu, Lizhen Cui, Di Wang, Jingfeng Zhang, Mohan Kankanhalli

**Abstract**: The wide-ranging applications of large language models (LLMs), especially in safety-critical domains, necessitate the proper evaluation of the LLM's adversarial robustness. This paper proposes an efficient tool to audit the LLM's adversarial robustness via a prompt-based adversarial attack (PromptAttack). PromptAttack converts adversarial textual attacks into an attack prompt that can cause the victim LLM to output the adversarial sample to fool itself. The attack prompt is composed of three important components: (1) original input (OI) including the original sample and its ground-truth label, (2) attack objective (AO) illustrating a task description of generating a new sample that can fool itself without changing the semantic meaning, and (3) attack guidance (AG) containing the perturbation instructions to guide the LLM on how to complete the task by perturbing the original sample at character, word, and sentence levels, respectively. Besides, we use a fidelity filter to ensure that PromptAttack maintains the original semantic meanings of the adversarial examples. Further, we enhance the attack power of PromptAttack by ensembling adversarial examples at different perturbation levels. Comprehensive empirical results using Llama2 and GPT-3.5 validate that PromptAttack consistently yields a much higher attack success rate compared to AdvGLUE and AdvGLUE++. Interesting findings include that a simple emoji can easily mislead GPT-3.5 to make wrong predictions.

摘要: 大型语言模型(LLM)的广泛应用，特别是在安全关键领域，需要对LLM的攻击健壮性进行适当的评估。提出了一种通过基于提示的敌意攻击(PromptAttack)来审计LLM攻击健壮性的有效工具。PromptAttack将对抗性文本攻击转换为攻击提示，这可能会导致受害者LLM输出对抗性样本来愚弄自己。攻击提示由三个重要部分组成：(1)原始输入(OI)，包括原始样本及其地面事实标签；(2)攻击目标(AO)，说明生成可以在不改变语义的情况下欺骗自己的新样本的任务描述；(3)攻击指导(AG)，包含扰动指令，以分别在字、词和句子层面指导LLM如何通过扰动原始样本来完成任务。此外，我们使用了一个保真度过滤器来确保PromptAttack保持了对抗性例子的原始语义。此外，我们通过集成不同扰动级别的对抗性实例来增强PromptAttack的攻击能力。使用Llama2和GPT-3.5的综合实验结果验证了PromptAttack始终比AdvGLUE和AdvGLUE++产生更高的攻击成功率。有趣的发现包括，一个简单的表情符号很容易误导GPT-3.5做出错误的预测。



## **39. Mitigating Backdoor Poisoning Attacks through the Lens of Spurious Correlation**

通过伪关联镜头缓解后门中毒攻击 cs.CL

accepted to EMNLP2023 (main conference)

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2305.11596v2) [paper-pdf](http://arxiv.org/pdf/2305.11596v2)

**Authors**: Xuanli He, Qiongkai Xu, Jun Wang, Benjamin Rubinstein, Trevor Cohn

**Abstract**: Modern NLP models are often trained over large untrusted datasets, raising the potential for a malicious adversary to compromise model behaviour. For instance, backdoors can be implanted through crafting training instances with a specific textual trigger and a target label. This paper posits that backdoor poisoning attacks exhibit \emph{spurious correlation} between simple text features and classification labels, and accordingly, proposes methods for mitigating spurious correlation as means of defence. Our empirical study reveals that the malicious triggers are highly correlated to their target labels; therefore such correlations are extremely distinguishable compared to those scores of benign features, and can be used to filter out potentially problematic instances. Compared with several existing defences, our defence method significantly reduces attack success rates across backdoor attacks, and in the case of insertion-based attacks, our method provides a near-perfect defence.

摘要: 现代NLP模型通常是在不可信的大型数据集上进行训练的，这增加了恶意对手危害模型行为的可能性。例如，可以通过制作带有特定文本触发器和目标标签的训练实例来植入后门。文章假设后门中毒攻击在简单文本特征和分类标签之间表现出虚假相关性，并相应地提出了减少虚假相关性的方法作为防御手段。我们的经验研究表明，恶意触发器与其目标标签高度相关；因此，与那些良性特征相比，这种相关性非常容易区分，可以用来过滤潜在的问题实例。与现有的几种防御方法相比，我们的防御方法显著降低了后门攻击的攻击成功率，并且在基于插入的攻击中，我们的方法提供了近乎完美的防御。



## **40. Generalizable Lightweight Proxy for Robust NAS against Diverse Perturbations**

针对不同扰动的健壮NAS通用型轻量级代理 cs.LG

NeurIPS 2023, Code is available at  https://github.com/HyeonjeongHa/CRoZe

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2306.05031v2) [paper-pdf](http://arxiv.org/pdf/2306.05031v2)

**Authors**: Hyeonjeong Ha, Minseon Kim, Sung Ju Hwang

**Abstract**: Recent neural architecture search (NAS) frameworks have been successful in finding optimal architectures for given conditions (e.g., performance or latency). However, they search for optimal architectures in terms of their performance on clean images only, while robustness against various types of perturbations or corruptions is crucial in practice. Although there exist several robust NAS frameworks that tackle this issue by integrating adversarial training into one-shot NAS, however, they are limited in that they only consider robustness against adversarial attacks and require significant computational resources to discover optimal architectures for a single task, which makes them impractical in real-world scenarios. To address these challenges, we propose a novel lightweight robust zero-cost proxy that considers the consistency across features, parameters, and gradients of both clean and perturbed images at the initialization state. Our approach facilitates an efficient and rapid search for neural architectures capable of learning generalizable features that exhibit robustness across diverse perturbations. The experimental results demonstrate that our proxy can rapidly and efficiently search for neural architectures that are consistently robust against various perturbations on multiple benchmark datasets and diverse search spaces, largely outperforming existing clean zero-shot NAS and robust NAS with reduced search cost.

摘要: 最近的神经体系结构搜索(NAS)框架已经成功地找到了针对给定条件(例如，性能或延迟)的最佳体系结构。然而，他们只根据在干净图像上的性能来寻找最佳体系结构，而对各种类型的扰动或损坏的健壮性在实践中是至关重要的。虽然有几个健壮的NAS框架通过将对抗性训练集成到一次性NAS来解决这个问题，但是它们的局限性在于它们只考虑针对对抗性攻击的健壮性，并且需要大量的计算资源来发现单个任务的最佳架构，这使得它们在现实世界的场景中不切实际。为了应对这些挑战，我们提出了一种新的轻量级健壮零代价代理，该代理在初始化状态下考虑了干净图像和扰动图像的特征、参数和梯度的一致性。我们的方法有助于高效和快速地搜索能够学习在不同扰动下表现出健壮性的可概括特征的神经体系结构。实验结果表明，我们的代理能够快速有效地搜索到在多个基准数据集和不同搜索空间上对各种扰动具有一致健壮性的神经体系结构，大大优于现有的干净的零镜头NAS和健壮的NAS，并且降低了搜索成本。



## **41. Detecting Shared Data Manipulation in Distributed Optimization Algorithms**

分布式优化算法中共享数据操作的检测 eess.SY

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13252v1) [paper-pdf](http://arxiv.org/pdf/2310.13252v1)

**Authors**: Mohannad Alkhraijah, Rachel Harris, Samuel Litchfield, David Huggins, Daniel K. Molzahn

**Abstract**: This paper investigates the vulnerability of the Alternating Direction Method of Multipliers (ADMM) algorithm to shared data manipulation, with a focus on solving optimal power flow (OPF) problems. Deliberate data manipulation may cause the ADMM algorithm to converge to suboptimal solutions. We derive two sufficient conditions for detecting data manipulation based on the theoretical convergence trajectory of the ADMM algorithm. We evaluate the detection conditions' performance on three data manipulation strategies we previously proposed: simple, feedback, and bilevel optimization attacks. We then extend these three data manipulation strategies to avoid detection by considering both the detection conditions and a neural network (NN) detection model in the attacks. We also propose an adversarial NN training framework to detect shared data manipulation. We illustrate the performance of our data manipulation strategy and detection framework on OPF problems. The results show that the proposed detection conditions successfully detect most of the data manipulation attacks. However, a bilevel optimization attack strategy that incorporates the detection methods may avoid being detected. Countering this, our proposed adversarial training framework detects all the instances of the bilevel optimization attack.

摘要: 研究了交替方向乘子法(ADMM)算法对共享数据操作的脆弱性，重点研究了求解最优潮流(OPF)问题。故意的数据处理可能会导致ADMM算法收敛到次优解。基于ADMM算法的理论收敛轨迹，我们得到了检测数据操纵的两个充分条件。我们评估了检测条件在我们之前提出的三种数据操作策略上的性能：简单攻击、反馈攻击和双层优化攻击。然后，我们扩展了这三种数据操作策略，通过在攻击中同时考虑检测条件和神经网络检测模型来避免检测。我们还提出了一种对抗性神经网络训练框架来检测共享数据的篡改。我们举例说明了我们的数据操作策略和检测框架在最优潮流问题上的性能。实验结果表明，所提出的检测条件能够成功检测出大部分数据操纵攻击。然而，结合了检测方法的双层优化攻击策略可以避免被检测到。针对这一点，我们提出的对抗性训练框架检测到两级优化攻击的所有实例。



## **42. Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models**

朝向稳健剪枝：一种自适应的语言模型知识保留剪枝策略 cs.CL

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.13191v1) [paper-pdf](http://arxiv.org/pdf/2310.13191v1)

**Authors**: Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu

**Abstract**: The pruning objective has recently extended beyond accuracy and sparsity to robustness in language models. Despite this, existing methods struggle to enhance robustness against adversarial attacks when continually increasing model sparsity and require a retraining process. As humans step into the era of large language models, these issues become increasingly prominent. This paper proposes that the robustness of language models is proportional to the extent of pre-trained knowledge they encompass. Accordingly, we introduce a post-training pruning strategy designed to faithfully replicate the embedding space and feature space of dense language models, aiming to conserve more pre-trained knowledge during the pruning process. In this setup, each layer's reconstruction error not only originates from itself but also includes cumulative error from preceding layers, followed by an adaptive rectification. Compared to other state-of-art baselines, our approach demonstrates a superior balance between accuracy, sparsity, robustness, and pruning cost with BERT on datasets SST2, IMDB, and AGNews, marking a significant stride towards robust pruning in language models.

摘要: 修剪目标最近已经超越了语言模型中的精确度和稀疏性，扩展到了健壮性。尽管如此，现有的方法在不断增加模型稀疏性的同时努力增强对敌对攻击的鲁棒性，并且需要重新训练过程。随着人类步入大型语言模型时代，这些问题变得日益突出。本文提出语言模型的稳健性与它们所包含的预训练知识的程度成正比。因此，我们提出了一种训练后剪枝策略，旨在忠实地复制密集语言模型的嵌入空间和特征空间，目的是在剪枝过程中保存更多的预先训练的知识。在这种设置中，每一层的重建误差不仅源于自身，还包括来自前几层的累积误差，然后进行自适应校正。与其他最先进的基线相比，我们的方法在精确度、稀疏性、健壮性和剪枝成本之间表现出了更好的平衡，在数据集Sst2、IMDB和AgNews上使用ERT，标志着在语言模型中朝着健壮剪枝迈出了重要的一步。



## **43. Quantum Key Distribution for Critical Infrastructures: Towards Cyber Physical Security for Hydropower and Dams**

关键基础设施的量子密钥分发：迈向水电和大坝的网络物理安全 quant-ph

20 pages, 7 figures, 6 tables

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.13100v1) [paper-pdf](http://arxiv.org/pdf/2310.13100v1)

**Authors**: Adrien Green, Jeremy Lawrence, George Siopsis, Nicholas Peters, Ali Passian

**Abstract**: Hydropower facilities are often remotely monitored or controlled from a centralized remote-control room. Additionally, major component manufacturers monitor the performance of installed components. While these communications enable efficiencies and increased reliability, they also expand the cyber-attack surface. Communications may use the internet to remote control a facility's control systems, or it may involve sending control commands over a network from a control room to a machine. The content could be encrypted and decrypted using a public key to protect the communicated information. These cryptographic encoding and decoding schemes have been shown to be vulnerable, a situation which is being exacerbated as more advances are made in computer technologies such as quantum computing. In contrast, quantum key distribution (QKD) is not based upon a computational problem, and offers an alternative to conventional public-key cryptography. Although the underlying mechanism of QKD ensures that any attempt by an adversary to observe the quantum part of the protocol will result in a detectable signature as an increased error rate, potentially even preventing key generation, it serves as a warning for further investigation. When the error rate is low enough and enough photons have been detected, a shared private key can be generated known only to the sender and receiver. We describe how this novel technology and its several modalities could benefit the critical infrastructures of dams or hydropower facilities. The presented discussions may be viewed as a precursor to a quantum cybersecurity roadmap for the identification of relevant threats and mitigation.

摘要: 水电设施通常通过一个集中的远程控制室进行远程监控。此外，主要组件制造商还监控已安装组件的性能。虽然这些通信提高了效率和可靠性，但它们也扩大了网络攻击面。通信可以使用互联网来远程控制设施的控制系统，或者它可能涉及通过网络从控制室向机器发送控制命令。可以使用公钥对内容进行加密和解密，以保护传送的信息。这些加密编码和解码方案已被证明是脆弱的，随着量子计算等计算机技术的更多进步，这种情况正在加剧。相比之下，量子密钥分发(QKD)不是基于计算问题，而是提供了传统公钥密码术的替代方案。虽然QKD的基本机制确保了攻击者观察协议量子部分的任何尝试都会导致可检测的签名，因为这会增加错误率，甚至可能阻止密钥生成，但它可以作为进一步调查的警告。当误码率足够低并且已经检测到足够多的光子时，可以生成只有发送者和接收者知道的共享私钥。我们描述了这项新技术及其几种形式如何使大坝或水电设施的关键基础设施受益。所提出的讨论可被视为确定相关威胁和缓解的量子网络安全路线图的先驱。



## **44. PatchCURE: Improving Certifiable Robustness, Model Utility, and Computation Efficiency of Adversarial Patch Defenses**

PatchCURE：改进对抗性补丁防御的可证明的健壮性、模型实用性和计算效率 cs.CV

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.13076v1) [paper-pdf](http://arxiv.org/pdf/2310.13076v1)

**Authors**: Chong Xiang, Tong Wu, Sihui Dai, Jonathan Petit, Suman Jana, Prateek Mittal

**Abstract**: State-of-the-art defenses against adversarial patch attacks can now achieve strong certifiable robustness with a marginal drop in model utility. However, this impressive performance typically comes at the cost of 10-100x more inference-time computation compared to undefended models -- the research community has witnessed an intense three-way trade-off between certifiable robustness, model utility, and computation efficiency. In this paper, we propose a defense framework named PatchCURE to approach this trade-off problem. PatchCURE provides sufficient "knobs" for tuning defense performance and allows us to build a family of defenses: the most robust PatchCURE instance can match the performance of any existing state-of-the-art defense (without efficiency considerations); the most efficient PatchCURE instance has similar inference efficiency as undefended models. Notably, PatchCURE achieves state-of-the-art robustness and utility performance across all different efficiency levels, e.g., 16-23% absolute clean accuracy and certified robust accuracy advantages over prior defenses when requiring computation efficiency to be close to undefended models. The family of PatchCURE defenses enables us to flexibly choose appropriate defenses to satisfy given computation and/or utility constraints in practice.

摘要: 针对对抗性补丁攻击的最先进防御现在可以实现强大的可证明的健壮性，同时模型效用略有下降。然而，这种令人印象深刻的性能通常是以比无防御模型多10-100倍的推理时间计算为代价的--研究界见证了可证明的健壮性、模型实用性和计算效率之间的激烈三方权衡。在本文中，我们提出了一个名为PatchCURE的防御框架来解决这个权衡问题。PatchCURE为调整防御性能提供了足够的“旋钮”，并允许我们构建一系列防御：最健壮的PatchCURE实例可以与任何现有最先进的防御实例的性能相媲美(无需考虑效率)；最高效的PatchCURE实例具有与无防御模型相似的推理效率。值得注意的是，PatchCURE在所有不同的效率水平上实现了最先进的稳健性和实用性能，例如，当需要计算效率接近无防御模型时，绝对清洁准确率为16%-23%，并且经过认证的稳健精确度优于以前的防御系统。PatchCURE防御体系使我们能够灵活地选择适当的防御，以满足实践中给定的计算和/或效用约束。



## **45. Categorical composable cryptography: extended version**

范畴可合成密码学：扩展版本 cs.CR

Extended version of arXiv:2105.05949 which appeared in FoSSaCS 2022.  Very minor layout updates to the previous version as requested by the journal

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2208.13232v3) [paper-pdf](http://arxiv.org/pdf/2208.13232v3)

**Authors**: Anne Broadbent, Martti Karvonen

**Abstract**: We formalize the simulation paradigm of cryptography in terms of category theory and show that protocols secure against abstract attacks form a symmetric monoidal category, thus giving an abstract model of composable security definitions in cryptography. Our model is able to incorporate computational security, set-up assumptions and various attack models such as colluding or independently acting subsets of adversaries in a modular, flexible fashion. We conclude by using string diagrams to rederive the security of the one-time pad, correctness of Diffie-Hellman key exchange and no-go results concerning the limits of bipartite and tripartite cryptography, ruling out e.g., composable commitments and broadcasting. On the way, we exhibit two categorical constructions of resource theories that might be of independent interest: one capturing resources shared among multiple parties and one capturing resource conversions that succeed asymptotically.

摘要: 我们用范畴理论形式化了密码学的模拟范型，证明了对抽象攻击安全的协议形成了对称的么半范畴，从而给出了密码学中可组合安全定义的抽象模型。我们的模型能够以模块化、灵活的方式结合计算安全性、设置假设和各种攻击模型，例如串通或独立行动的对手子集。最后，我们使用字符串图重新推导了一次性密钥的安全性，Diffie-Hellman密钥交换的正确性，以及关于二方和三方密码术限制的不可行结果，排除了例如可组合承诺和广播。在此过程中，我们展示了两种可能独立感兴趣的资源理论范畴结构：一种是捕获多方共享的资源，另一种是捕获渐近成功的资源转换。



## **46. OODRobustBench: benchmarking and analyzing adversarial robustness under distribution shift**

OODRobustBch：分布转移下的对手健壮性基准测试与分析 cs.LG

in submission

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12793v1) [paper-pdf](http://arxiv.org/pdf/2310.12793v1)

**Authors**: Lin Li, Yifei Wang, Chawin Sitawarin, Michael Spratling

**Abstract**: Existing works have made great progress in improving adversarial robustness, but typically test their method only on data from the same distribution as the training data, i.e. in-distribution (ID) testing. As a result, it is unclear how such robustness generalizes under input distribution shifts, i.e. out-of-distribution (OOD) testing. This is a concerning omission as such distribution shifts are unavoidable when methods are deployed in the wild. To address this issue we propose a benchmark named OODRobustBench to comprehensively assess OOD adversarial robustness using 23 dataset-wise shifts (i.e. naturalistic shifts in input distribution) and 6 threat-wise shifts (i.e., unforeseen adversarial threat models). OODRobustBench is used to assess 706 robust models using 60.7K adversarial evaluations. This large-scale analysis shows that: 1) adversarial robustness suffers from a severe OOD generalization issue; 2) ID robustness correlates strongly with OOD robustness, in a positive linear way, under many distribution shifts. The latter enables the prediction of OOD robustness from ID robustness. Based on this, we are able to predict the upper limit of OOD robustness for existing robust training schemes. The results suggest that achieving OOD robustness requires designing novel methods beyond the conventional ones. Last, we discover that extra data, data augmentation, advanced model architectures and particular regularization approaches can improve OOD robustness. Noticeably, the discovered training schemes, compared to the baseline, exhibit dramatically higher robustness under threat shift while keeping high ID robustness, demonstrating new promising solutions for robustness against both multi-attack and unforeseen attacks.

摘要: 现有的工作在提高对手的稳健性方面已经取得了很大的进展，但通常只在来自与训练数据相同的分布的数据上测试他们的方法，即内分布(ID)测试。因此，目前还不清楚这种稳健性如何在输入分布漂移(即分布外(OOD)测试)下得到推广。这是一个令人担忧的遗漏，因为当方法部署在野外时，这种分布变化是不可避免的。为了解决这一问题，我们提出了一个名为OODRobustBch的基准，该基准使用23个数据集方向的变化(即输入分布的自然变化)和6个威胁方向的变化(即不可预见的对手威胁模型)来综合评估OOD对手威胁的健壮性。使用60.7K的对抗性评估，OODRobustBch被用来评估706个稳健模型。这一大规模的分析表明：1)对手的健壮性受到严重的OOD泛化问题的影响；2)ID健壮性与OOD健壮性在许多分布变化下以正线性的方式强烈相关。后者使得能够从ID健壮性预测OOD健壮性。在此基础上，我们能够预测现有健壮训练方案的OOD稳健性上限。结果表明，要实现面向对象设计的健壮性，需要设计出超越传统方法的新方法。最后，我们发现额外的数据、数据扩充、先进的模型结构和特定的正则化方法可以提高面向对象设计的健壮性。值得注意的是，与基线相比，发现的训练方案在保持高ID稳健性的同时，在威胁转移下表现出显著更高的稳健性，展示了针对多攻击和不可预见攻击的稳健性的新的有前途的解决方案。



## **47. TorKameleon: Improving Tor's Censorship Resistance with K-anonymization and Media-based Covert Channels**

TorKameleon：通过K-匿名化和基于媒体的隐蔽渠道提高Tor的审查抵抗力 cs.CR

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2303.17544v3) [paper-pdf](http://arxiv.org/pdf/2303.17544v3)

**Authors**: Afonso Vilalonga, João S. Resende, Henrique Domingos

**Abstract**: Anonymity networks like Tor significantly enhance online privacy but are vulnerable to correlation attacks by state-level adversaries. While covert channels encapsulated in media protocols, particularly WebRTC-based encapsulation, have demonstrated effectiveness against passive traffic correlation attacks, their resilience against active correlation attacks remains unexplored, and their compatibility with Tor has been limited. This paper introduces TorKameleon, a censorship evasion solution designed to protect Tor users from both passive and active correlation attacks. TorKameleon employs K-anonymization techniques to fragment and reroute traffic through multiple TorKameleon proxies, while also utilizing covert WebRTC-based channels or TLS tunnels to encapsulate user traffic.

摘要: 像Tor这样的匿名网络极大地增强了在线隐私，但容易受到国家级对手的关联攻击。虽然封装在媒体协议中的隐蔽信道，特别是基于WebRTC的封装，已经证明了对被动流量关联攻击的有效性，但它们对主动关联攻击的弹性还没有被探索，并且它们与ToR的兼容性已经受到限制。本文介绍了TorKameleon，一个旨在保护Tor用户免受被动和主动相关攻击的审查逃避解决方案。TorKameleon使用K-匿名技术通过多个TorKameleon代理对流量进行分段和重新路由，同时还利用基于WebRTC的隐蔽通道或TLS隧道来封装用户流量。



## **48. WaveAttack: Asymmetric Frequency Obfuscation-based Backdoor Attacks Against Deep Neural Networks**

WaveAttack：基于非对称频率混淆的深层神经网络后门攻击 cs.CV

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.11595v2) [paper-pdf](http://arxiv.org/pdf/2310.11595v2)

**Authors**: Jun Xia, Zhihao Yue, Yingbo Zhou, Zhiwei Ling, Xian Wei, Mingsong Chen

**Abstract**: Due to the popularity of Artificial Intelligence (AI) technology, numerous backdoor attacks are designed by adversaries to mislead deep neural network predictions by manipulating training samples and training processes. Although backdoor attacks are effective in various real scenarios, they still suffer from the problems of both low fidelity of poisoned samples and non-negligible transfer in latent space, which make them easily detectable by existing backdoor detection algorithms. To overcome the weakness, this paper proposes a novel frequency-based backdoor attack method named WaveAttack, which obtains image high-frequency features through Discrete Wavelet Transform (DWT) to generate backdoor triggers. Furthermore, we introduce an asymmetric frequency obfuscation method, which can add an adaptive residual in the training and inference stage to improve the impact of triggers and further enhance the effectiveness of WaveAttack. Comprehensive experimental results show that WaveAttack not only achieves higher stealthiness and effectiveness, but also outperforms state-of-the-art (SOTA) backdoor attack methods in the fidelity of images by up to 28.27\% improvement in PSNR, 1.61\% improvement in SSIM, and 70.59\% reduction in IS.

摘要: 由于人工智能(AI)技术的普及，许多后门攻击都是由对手设计的，通过操纵训练样本和训练过程来误导深度神经网络预测。虽然后门攻击在各种真实场景中都是有效的，但它们仍然存在有毒样本保真度低和潜在空间传输不可忽略的问题，这使得它们很容易被现有的后门检测算法检测到。针对这一缺陷，提出了一种新的基于频率的后门攻击方法WaveAttack，该方法通过离散小波变换(DWT)提取图像高频特征来生成后门触发器。此外，我们还引入了一种非对称频率混淆方法，在训练和推理阶段加入自适应残差，以改善触发的影响，进一步增强WaveAttack的有效性。综合实验结果表明，WaveAttack不仅获得了更高的隐蔽性和有效性，而且在图像保真度方面优于最新的SOTA后门攻击方法，峰值信噪比提高了28.27，SSIM提高了1.61，IS降低了70.59。



## **49. Learn from the Past: A Proxy based Adversarial Defense Framework to Boost Robustness**

借鉴过去：一种基于代理的增强健壮性的对抗防御框架 cs.LG

16 Pages

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12713v1) [paper-pdf](http://arxiv.org/pdf/2310.12713v1)

**Authors**: Yaohua Liu, Jiaxin Gao, Zhu Liu, Xianghao Jiao, Xin Fan, Risheng Liu

**Abstract**: In light of the vulnerability of deep learning models to adversarial samples and the ensuing security issues, a range of methods, including Adversarial Training (AT) as a prominent representative, aimed at enhancing model robustness against various adversarial attacks, have seen rapid development. However, existing methods essentially assist the current state of target model to defend against parameter-oriented adversarial attacks with explicit or implicit computation burdens, which also suffers from unstable convergence behavior due to inconsistency of optimization trajectories. Diverging from previous work, this paper reconsiders the update rule of target model and corresponding deficiency to defend based on its current state. By introducing the historical state of the target model as a proxy, which is endowed with much prior information for defense, we formulate a two-stage update rule, resulting in a general adversarial defense framework, which we refer to as `LAST' ({\bf L}earn from the P{\bf ast}). Besides, we devise a Self Distillation (SD) based defense objective to constrain the update process of the proxy model without the introduction of larger teacher models. Experimentally, we demonstrate consistent and significant performance enhancements by refining a series of single-step and multi-step AT methods (e.g., up to $\bf 9.2\%$ and $\bf 20.5\%$ improvement of Robust Accuracy (RA) on CIFAR10 and CIFAR100 datasets, respectively) across various datasets, backbones and attack modalities, and validate its ability to enhance training stability and ameliorate catastrophic overfitting issues meanwhile.

摘要: 鉴于深度学习模型对对抗性样本的脆弱性以及随之而来的安全问题，旨在增强模型对各种对抗性攻击的稳健性的一系列方法，包括作为突出代表的对抗性训练(AT)，得到了迅速的发展。然而，现有的方法本质上是帮助目标模型的当前状态来防御具有显式或隐式计算负担的面向参数的对抗性攻击，而这种对抗性攻击也存在由于优化轨迹不一致而导致的不稳定收敛行为。与前人的工作不同，本文针对目标模型的现状，重新考虑了目标模型的更新规则以及相应的不足进行防御。通过引入目标模型的历史状态作为代理，赋予目标模型大量的先验信息用于防御，我们制定了一个两阶段更新规则，从而得到了一个通用的对抗性防御框架，我们称之为`last‘(L从P{\bf ast}中赚取)。此外，我们设计了一个基于自蒸馏(SD)的防御目标来约束代理模型的更新过程，而不需要引入更大的教师模型。在实验上，我们通过提炼一系列单步和多步AT方法(例如，分别在CIFAR10和CIFAR100数据集上提高稳健精度(RA)，分别高达9.2美元和20.5美元)，展示了持续和显著的性能提升，并验证了其增强训练稳定性和改善灾难性过拟合问题的能力。



## **50. Generating Robust Adversarial Examples against Online Social Networks (OSNs)**

生成针对在线社交网络(OSN)的强大敌意示例 cs.MM

26 pages, 9 figures

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12708v1) [paper-pdf](http://arxiv.org/pdf/2310.12708v1)

**Authors**: Jun Liu, Jiantao Zhou, Haiwei Wu, Weiwei Sun, Jinyu Tian

**Abstract**: Online Social Networks (OSNs) have blossomed into prevailing transmission channels for images in the modern era. Adversarial examples (AEs) deliberately designed to mislead deep neural networks (DNNs) are found to be fragile against the inevitable lossy operations conducted by OSNs. As a result, the AEs would lose their attack capabilities after being transmitted over OSNs. In this work, we aim to design a new framework for generating robust AEs that can survive the OSN transmission; namely, the AEs before and after the OSN transmission both possess strong attack capabilities. To this end, we first propose a differentiable network termed SImulated OSN (SIO) to simulate the various operations conducted by an OSN. Specifically, the SIO network consists of two modules: 1) a differentiable JPEG layer for approximating the ubiquitous JPEG compression and 2) an encoder-decoder subnetwork for mimicking the remaining operations. Based upon the SIO network, we then formulate an optimization framework to generate robust AEs by enforcing model outputs with and without passing through the SIO to be both misled. Extensive experiments conducted over Facebook, WeChat and QQ demonstrate that our attack methods produce more robust AEs than existing approaches, especially under small distortion constraints; the performance gain in terms of Attack Success Rate (ASR) could be more than 60%. Furthermore, we build a public dataset containing more than 10,000 pairs of AEs processed by Facebook, WeChat or QQ, facilitating future research in the robust AEs generation. The dataset and code are available at https://github.com/csjunjun/RobustOSNAttack.git.

摘要: 在现代，在线社交网络(OSN)已经发展成为流行的图像传输渠道。对抗性例子(AE)被发现被故意设计来误导深度神经网络(DNN)，对OSN进行的不可避免的有损操作是脆弱的。因此，在通过OSN传输后，AE将失去攻击能力。在这项工作中，我们的目标是设计一种新的框架来生成能够在OSN传输中幸存下来的健壮的AE，即OSN传输前后的AE都具有很强的攻击能力。为此，我们首先提出了一种称为模拟OSN(SIO)的可区分网络来模拟OSN进行的各种操作。具体地说，SIO网络由两个模块组成：1)用于近似普遍存在的JPEG压缩的可区分JPEG层和2)用于模拟其余操作的编解码子网络。然后，基于SIO网络，我们制定了一个优化框架，通过强制模型输出在通过和不通过SIO的情况下都被误导来生成稳健的AE。在脸书、微信和QQ上进行的大量实验表明，我们的攻击方法比现有的方法产生了更强的攻击效果，特别是在小失真约束下；攻击成功率方面的性能增益可以超过60%。此外，我们还建立了一个公共数据集，其中包含了超过10,000对由脸书、微信或QQ处理的实体实体，为未来稳健的实体实体的研究提供了便利。数据集和代码可在https://github.com/csjunjun/RobustOSNAttack.git.上获得



