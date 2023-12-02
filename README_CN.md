# Latest Adversarial Attack Papers
**update at 2023-12-02 11:30:07**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Attacks and Defenses for Wireless Signal Classifiers using CDI-aware GANs**

基于CDI感知Gans的无线信号分类器的对抗性攻击与防御 cs.IT

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.18820v1) [paper-pdf](http://arxiv.org/pdf/2311.18820v1)

**Authors**: Sujata Sinha, Alkan Soysal

**Abstract**: We introduce a Channel Distribution Information (CDI)-aware Generative Adversarial Network (GAN), designed to address the unique challenges of adversarial attacks in wireless communication systems. The generator in this CDI-aware GAN maps random input noise to the feature space, generating perturbations intended to deceive a target modulation classifier. Its discriminators play a dual role: one enforces that the perturbations follow a Gaussian distribution, making them indistinguishable from Gaussian noise, while the other ensures these perturbations account for realistic channel effects and resemble no-channel perturbations.   Our proposed CDI-aware GAN can be used as an attacker and a defender. In attack scenarios, the CDI-aware GAN demonstrates its prowess by generating robust adversarial perturbations that effectively deceive the target classifier, outperforming known methods. Furthermore, CDI-aware GAN as a defender significantly improves the target classifier's resilience against adversarial attacks.

摘要: 我们介绍了一种基于信道分布信息(CDI)的生成性对抗网络(GAN)，旨在解决无线通信系统中对抗攻击的独特挑战。这种CDI感知的GaN中的生成器将随机输入噪声映射到特征空间，生成旨在欺骗目标调制分类器的扰动。它的鉴别器扮演着双重角色：一个强制扰动服从高斯分布，使它们与高斯噪声无法区分，另一个确保这些扰动考虑到真实的信道影响并类似于无信道扰动。我们建议的CDI感知GAN可以用作攻击者和防御者。在攻击场景中，支持CDI的GAN通过生成强健的对抗性扰动来展示其能力，从而有效地欺骗目标分类器，性能优于已知的方法。此外，CDI感知的GAN作为防御者显著提高了目标分类器对对手攻击的弹性。



## **2. Differentiable JPEG: The Devil is in the Details**

上一篇：JPEG：魔鬼在细节中 cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2309.06978v3) [paper-pdf](http://arxiv.org/pdf/2309.06978v3)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.

摘要: JPEG仍然是应用最广泛的有损图像编码方法之一。然而，JPEG的不可微特性限制了其在深度学习管道中的应用。为了解决这个问题，最近已经提出了几种JPEG的可微近似。本文对现有的DIFF进行了全面的回顾。JPEG处理并确定了以前方法遗漏的关键细节。为此，我们提出了一个新颖的Diff。JPEG方法，克服了以前的限制。我们的方法是可微的W.r.t。输入图像、JPEG质量、量化表和颜色转换参数。我们评估了DIFF的向前和向后性能。JPEG方法与现有方法的对比。此外，还进行了广泛的消融，以评估关键的设计选择。我们提议的不同之处。JPEG与(Non-Diff.)参考实现最好，大大超过了最近最好的差异。平均接近3.47美元分贝(PSNR)。对于强压缩率，我们甚至可以将PSNR提高9.51美元分贝。强大的对抗性攻击结果是由我们的差异产生的。JPEG格式，演示了有效的渐变近似。我们的代码可以在https://github.com/necla-ml/Diff-JPEG.上找到



## **3. Diffusion Models for Imperceptible and Transferable Adversarial Attack**

不可察觉和可转移对抗性攻击的扩散模型 cs.CV

Code Page: https://github.com/WindVChen/DiffAttack. In Paper Version  v2, we incorporate more discussions and experiments

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2305.08192v2) [paper-pdf](http://arxiv.org/pdf/2305.08192v2)

**Authors**: Jianqi Chen, Hao Chen, Keyan Chen, Yilan Zhang, Zhengxia Zou, Zhenwei Shi

**Abstract**: Many existing adversarial attacks generate $L_p$-norm perturbations on image RGB space. Despite some achievements in transferability and attack success rate, the crafted adversarial examples are easily perceived by human eyes. Towards visual imperceptibility, some recent works explore unrestricted attacks without $L_p$-norm constraints, yet lacking transferability of attacking black-box models. In this work, we propose a novel imperceptible and transferable attack by leveraging both the generative and discriminative power of diffusion models. Specifically, instead of direct manipulation in pixel space, we craft perturbations in the latent space of diffusion models. Combined with well-designed content-preserving structures, we can generate human-insensitive perturbations embedded with semantic clues. For better transferability, we further "deceive" the diffusion model which can be viewed as an implicit recognition surrogate, by distracting its attention away from the target regions. To our knowledge, our proposed method, DiffAttack, is the first that introduces diffusion models into the adversarial attack field. Extensive experiments on various model structures, datasets, and defense methods have demonstrated the superiority of our attack over the existing attack methods.

摘要: 许多现有的对抗性攻击在图像RGB空间上产生$L_p$-范数扰动。尽管在可转移性和攻击成功率方面取得了一些成就，但制作的对抗性例子很容易被人眼察觉。对于视觉不可感知性，最近的一些工作探索了没有$L_p$-范数约束的无限攻击，但缺乏攻击黑盒模型的可转移性。在这项工作中，我们提出了一种新的不可察觉和可转移的攻击，利用扩散模型的生成性和区分性。具体地说，我们不是在像素空间中直接操作，而是在扩散模型的潜在空间中制造扰动。与设计良好的内容保持结构相结合，我们可以生成嵌入语义线索的人类不敏感的扰动。为了获得更好的可转移性，我们通过将扩散模型的注意力从目标区域转移开，进一步欺骗了可以被视为隐式识别代理的扩散模型。据我们所知，我们提出的DiffAttack方法首次将扩散模型引入到对抗性攻击领域。在各种模型结构、数据集和防御方法上的广泛实验证明了该攻击相对于现有攻击方法的优越性。



## **4. Data-Agnostic Model Poisoning against Federated Learning: A Graph Autoencoder Approach**

数据不可知模型毒化联合学习：一种图自动编码器方法 cs.LG

15 pages, 10 figures, submitted to IEEE Transactions on Information  Forensics and Security (TIFS)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.18498v1) [paper-pdf](http://arxiv.org/pdf/2311.18498v1)

**Authors**: Kai Li, Jingjing Zheng, Xin Yuan, Wei Ni, Ozgur B. Akan, H. Vincent Poor

**Abstract**: This paper proposes a novel, data-agnostic, model poisoning attack on Federated Learning (FL), by designing a new adversarial graph autoencoder (GAE)-based framework. The attack requires no knowledge of FL training data and achieves both effectiveness and undetectability. By listening to the benign local models and the global model, the attacker extracts the graph structural correlations among the benign local models and the training data features substantiating the models. The attacker then adversarially regenerates the graph structural correlations while maximizing the FL training loss, and subsequently generates malicious local models using the adversarial graph structure and the training data features of the benign ones. A new algorithm is designed to iteratively train the malicious local models using GAE and sub-gradient descent. The convergence of FL under attack is rigorously proved, with a considerably large optimality gap. Experiments show that the FL accuracy drops gradually under the proposed attack and existing defense mechanisms fail to detect it. The attack can give rise to an infection across all benign devices, making it a serious threat to FL.

摘要: 通过设计一种新的基于对抗性图自动编码器(GAE)的框架，提出了一种针对联邦学习(FL)的数据不可知的模型中毒攻击。该攻击不需要了解FL训练数据，并且实现了有效性和不可检测性。通过监听良性局部模型和全局模型，攻击者提取良性局部模型和证实模型的训练数据特征之间的图结构相关性。然后攻击者在最大化FL训练损失的同时对抗性地重新生成图结构相关性，并随后利用对抗性图结构和良性图结构的训练数据特征来生成恶意局部模型。设计了一种利用GAE和次梯度下降迭代训练恶意局部模型的新算法。严格地证明了FL在攻击下的收敛，但存在相当大的最优性差距。实验表明，在所提出的攻击下，FL的准确率逐渐下降，现有的防御机制无法检测到它。这种攻击可以引起对所有良性设备的感染，使其成为对FL的严重威胁。



## **5. Towards Safer Generative Language Models: A Survey on Safety Risks, Evaluations, and Improvements**

走向更安全的生成性语言模型：安全风险、评估和改进的综述 cs.AI

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2302.09270v3) [paper-pdf](http://arxiv.org/pdf/2302.09270v3)

**Authors**: Jiawen Deng, Jiale Cheng, Hao Sun, Zhexin Zhang, Minlie Huang

**Abstract**: As generative large model capabilities advance, safety concerns become more pronounced in their outputs. To ensure the sustainable growth of the AI ecosystem, it's imperative to undertake a holistic evaluation and refinement of associated safety risks. This survey presents a framework for safety research pertaining to large models, delineating the landscape of safety risks as well as safety evaluation and improvement methods. We begin by introducing safety issues of wide concern, then delve into safety evaluation methods for large models, encompassing preference-based testing, adversarial attack approaches, issues detection, and other advanced evaluation methods. Additionally, we explore the strategies for enhancing large model safety from training to deployment, highlighting cutting-edge safety approaches for each stage in building large models. Finally, we discuss the core challenges in advancing towards more responsible AI, including the interpretability of safety mechanisms, ongoing safety issues, and robustness against malicious attacks. Through this survey, we aim to provide clear technical guidance for safety researchers and encourage further study on the safety of large models.

摘要: 随着产生式大型模型能力的进步，安全问题在其输出中变得更加明显。为了确保人工智能生态系统的可持续增长，必须对相关安全风险进行全面评估和细化。本调查提出了与大型模型相关的安全研究框架，描绘了安全风险的图景以及安全评估和改进方法。我们首先介绍广泛关注的安全问题，然后深入研究大型模型的安全评估方法，包括基于偏好的测试、对抗性攻击方法、问题检测和其他高级评估方法。此外，我们还探讨了从培训到部署增强大型模型安全性的策略，重点介绍了构建大型模型的每个阶段的前沿安全方法。最后，我们讨论了向更负责任的人工智能发展的核心挑战，包括安全机制的可解释性、持续的安全问题和针对恶意攻击的健壮性。通过这次调查，我们旨在为安全研究人员提供明确的技术指导，并鼓励进一步研究大型模型的安全性。



## **6. On the Robustness of Decision-Focused Learning**

决策聚焦学习的稳健性研究 cs.LG

17 pages, 45 figures, submitted to AAAI artificial intelligence for  operations research workshop

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.16487v2) [paper-pdf](http://arxiv.org/pdf/2311.16487v2)

**Authors**: Yehya Farhat

**Abstract**: Decision-Focused Learning (DFL) is an emerging learning paradigm that tackles the task of training a machine learning (ML) model to predict missing parameters of an incomplete optimization problem, where the missing parameters are predicted. DFL trains an ML model in an end-to-end system, by integrating the prediction and optimization tasks, providing better alignment of the training and testing objectives. DFL has shown a lot of promise and holds the capacity to revolutionize decision-making in many real-world applications. However, very little is known about the performance of these models under adversarial attacks. We adopt ten unique DFL methods and benchmark their performance under two distinctly focused attacks adapted towards the Predict-then-Optimize problem setting. Our study proposes the hypothesis that the robustness of a model is highly correlated with its ability to find predictions that lead to optimal decisions without deviating from the ground-truth label. Furthermore, we provide insight into how to target the models that violate this condition and show how these models respond differently depending on the achieved optimality at the end of their training cycles.

摘要: 聚焦决策学习(DFL)是一种新兴的学习范式，它解决了训练机器学习(ML)模型来预测不完全优化问题的缺失参数的任务，其中缺失的参数被预测。DFL通过集成预测和优化任务，在端到端系统中训练ML模型，提供更好的训练和测试目标的一致性。DFL已经显示出了很大的潜力，并拥有在许多现实世界应用程序中彻底改变决策的能力。然而，人们对这些模型在对抗性攻击下的性能知之甚少。我们采用了十种独特的DFL方法，并对它们在两种针对预测-然后优化问题设置的明显集中的攻击下的性能进行了基准测试。我们的研究提出了这样的假设，即模型的稳健性与其在不偏离地面事实标签的情况下找到导致最优决策的预测的能力高度相关。此外，我们还提供了对如何针对违反这一条件的模型的洞察，并展示了这些模型如何根据在其训练周期结束时实现的最优化而做出不同的反应。



## **7. Improving the Robustness of Transformer-based Large Language Models with Dynamic Attention**

利用动态注意提高基于Transformer的大语言模型的健壮性 cs.CL

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.17400v2) [paper-pdf](http://arxiv.org/pdf/2311.17400v2)

**Authors**: Lujia Shen, Yuwen Pu, Shouling Ji, Changjiang Li, Xuhong Zhang, Chunpeng Ge, Ting Wang

**Abstract**: Transformer-based models, such as BERT and GPT, have been widely adopted in natural language processing (NLP) due to their exceptional performance. However, recent studies show their vulnerability to textual adversarial attacks where the model's output can be misled by intentionally manipulating the text inputs. Despite various methods that have been proposed to enhance the model's robustness and mitigate this vulnerability, many require heavy consumption resources (e.g., adversarial training) or only provide limited protection (e.g., defensive dropout). In this paper, we propose a novel method called dynamic attention, tailored for the transformer architecture, to enhance the inherent robustness of the model itself against various adversarial attacks. Our method requires no downstream task knowledge and does not incur additional costs. The proposed dynamic attention consists of two modules: (I) attention rectification, which masks or weakens the attention value of the chosen tokens, and (ii) dynamic modeling, which dynamically builds the set of candidate tokens. Extensive experiments demonstrate that dynamic attention significantly mitigates the impact of adversarial attacks, improving up to 33\% better performance than previous methods against widely-used adversarial attacks. The model-level design of dynamic attention enables it to be easily combined with other defense methods (e.g., adversarial training) to further enhance the model's robustness. Furthermore, we demonstrate that dynamic attention preserves the state-of-the-art robustness space of the original model compared to other dynamic modeling methods.

摘要: 基于变换的模型，如BERT和GPT，由于其优异的性能，在自然语言处理(NLP)中被广泛采用。然而，最近的研究表明，它们在文本对抗攻击中的脆弱性，其中模型的输出可能会被故意操纵文本输入误导。尽管已经提出了各种方法来增强模型的健壮性并缓解这一漏洞，但许多方法需要大量的资源(例如，对抗性训练)或仅提供有限的保护(例如，防御性退出)。在本文中，我们提出了一种新的方法，称为动态注意，为变压器结构量身定做，以增强模型本身对各种敌意攻击的内在稳健性。我们的方法不需要下游任务知识，也不会产生额外的成本。该算法由两个模块组成：(I)注意力纠正模块，用于屏蔽或削弱所选标记的关注值；(Ii)动态建模模块，用于动态构建候选标记集。大量实验表明，动态注意显著缓解了对抗性攻击的影响，比以往的方法在对抗广泛使用的对抗性攻击时的性能提高了33%。动态注意的模型级设计使得它可以很容易地与其他防御方法(如对抗性训练)相结合，进一步增强模型的稳健性。此外，与其他动态建模方法相比，动态注意保留了原始模型的最新稳健性空间。



## **8. Effective Backdoor Mitigation Depends on the Pre-training Objective**

有效的后门缓解取决于预训练目标 cs.LG

Accepted for oral presentation at BUGS workshop @ NeurIPS 2023  (https://neurips2023-bugs.github.io/)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.14948v2) [paper-pdf](http://arxiv.org/pdf/2311.14948v2)

**Authors**: Sahil Verma, Gantavya Bhatt, Avi Schwarzschild, Soumye Singhal, Arnav Mohanty Das, Chirag Shah, John P Dickerson, Jeff Bilmes

**Abstract**: Despite the advanced capabilities of contemporary machine learning (ML) models, they remain vulnerable to adversarial and backdoor attacks. This vulnerability is particularly concerning in real-world deployments, where compromised models may exhibit unpredictable behavior in critical scenarios. Such risks are heightened by the prevalent practice of collecting massive, internet-sourced datasets for pre-training multimodal models, as these datasets may harbor backdoors. Various techniques have been proposed to mitigate the effects of backdooring in these models such as CleanCLIP which is the current state-of-the-art approach. In this work, we demonstrate that the efficacy of CleanCLIP in mitigating backdoors is highly dependent on the particular objective used during model pre-training. We observe that stronger pre-training objectives correlate with harder to remove backdoors behaviors. We show this by training multimodal models on two large datasets consisting of 3 million (CC3M) and 6 million (CC6M) datapoints, under various pre-training objectives, followed by poison removal using CleanCLIP. We find that CleanCLIP is ineffective when stronger pre-training objectives are used, even with extensive hyperparameter tuning. Our findings underscore critical considerations for ML practitioners who pre-train models using large-scale web-curated data and are concerned about potential backdoor threats. Notably, our results suggest that simpler pre-training objectives are more amenable to effective backdoor removal. This insight is pivotal for practitioners seeking to balance the trade-offs between using stronger pre-training objectives and security against backdoor attacks.

摘要: 尽管当代机器学习(ML)模型具有先进的能力，但它们仍然容易受到对手和后门攻击。此漏洞在实际部署中尤其令人担忧，在实际部署中，受危害的模型可能会在关键情况下表现出不可预测的行为。为训练前的多模式模型收集来自互联网的海量数据集的普遍做法加剧了这种风险，因为这些数据集可能有后门。已经提出了各种技术来减轻这些模型中回溯的影响，例如CleanCLIP，这是当前最先进的方法。在这项工作中，我们证明了CleanCLIP在缓解后门方面的有效性高度依赖于在模型预培训期间使用的特定目标。我们观察到，较强的培训前目标与较难消除后门行为相关。我们通过在两个由300万(CC3M)和600万(CC6M)数据点组成的大型数据集上训练多模模型，在不同的预训练目标下，然后使用CleanCLIP去除毒物来证明这一点。我们发现，当使用更强的预培训目标时，即使进行了广泛的超参数调整，CleanCLIP也是无效的。我们的发现强调了ML从业者的关键考虑，他们使用大规模的网络管理数据对模型进行预培训，并担心潜在的后门威胁。值得注意的是，我们的结果表明，简单的预培训目标更容易有效地移除后门。对于寻求在使用更强的预培训目标和针对后门攻击的安全性之间进行权衡的从业者来说，这一见解至关重要。



## **9. AnonPSI: An Anonymity Assessment Framework for PSI**

AnonPSI：一个面向PSI的安全性评估框架 cs.CR

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.18118v1) [paper-pdf](http://arxiv.org/pdf/2311.18118v1)

**Authors**: Bo Jiang, Jian Du, Qiang Yan

**Abstract**: Private Set Intersection (PSI) is a widely used protocol that enables two parties to securely compute a function over the intersected part of their shared datasets and has been a significant research focus over the years. However, recent studies have highlighted its vulnerability to Set Membership Inference Attacks (SMIA), where an adversary might deduce an individual's membership by invoking multiple PSI protocols. This presents a considerable risk, even in the most stringent versions of PSI, which only return the cardinality of the intersection. This paper explores the evaluation of anonymity within the PSI context. Initially, we highlight the reasons why existing works fall short in measuring privacy leakage, and subsequently propose two attack strategies that address these deficiencies. Furthermore, we provide theoretical guarantees on the performance of our proposed methods. In addition to these, we illustrate how the integration of auxiliary information, such as the sum of payloads associated with members of the intersection (PSI-SUM), can enhance attack efficiency. We conducted a comprehensive performance evaluation of various attack strategies proposed utilizing two real datasets. Our findings indicate that the methods we propose markedly enhance attack efficiency when contrasted with previous research endeavors. {The effective attacking implies that depending solely on existing PSI protocols may not provide an adequate level of privacy assurance. It is recommended to combine privacy-enhancing technologies synergistically to enhance privacy protection even further.

摘要: 私有集合交集(PSI)是一种广泛使用的协议，它使双方能够安全地在其共享数据集的相交部分上计算函数，多年来一直是一个重要的研究热点。然而，最近的研究强调了它对集合成员推理攻击(SMIA)的脆弱性，在这种攻击中，攻击者可以通过调用多个PSI协议来推断个人的成员资格。这带来了相当大的风险，即使在最严格的PSI版本中也是如此，它只返回交集的基数。本文探讨了PSI环境下的匿名性评估。首先，我们强调了现有作品在测量隐私泄露方面不足的原因，并随后提出了两种攻击策略来解决这些不足。此外，我们还为我们所提出的方法的性能提供了理论保证。除此之外，我们还说明了辅助信息的集成，例如与交集成员相关的有效负载之和(PSI-SUM)如何提高攻击效率。我们利用两个真实数据集对提出的各种攻击策略进行了全面的性能评估。我们的研究结果表明，与以前的研究相比，我们提出的方法显着提高了攻击效率。{有效的攻击意味着，仅依赖现有的PSI协议可能无法提供足够级别的隐私保障。建议将增强隐私的技术协同结合，以进一步加强隐私保护。



## **10. Improving Faithfulness for Vision Transformers**

提高视觉变形金刚的忠诚度 cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17983v1) [paper-pdf](http://arxiv.org/pdf/2311.17983v1)

**Authors**: Lijie Hu, Yixin Liu, Ninghao Liu, Mengdi Huai, Lichao Sun, Di Wang

**Abstract**: Vision Transformers (ViTs) have achieved state-of-the-art performance for various vision tasks. One reason behind the success lies in their ability to provide plausible innate explanations for the behavior of neural architectures. However, ViTs suffer from issues with explanation faithfulness, as their focal points are fragile to adversarial attacks and can be easily changed with even slight perturbations on the input image. In this paper, we propose a rigorous approach to mitigate these issues by introducing Faithful ViTs (FViTs). Briefly speaking, an FViT should have the following two properties: (1) The top-$k$ indices of its self-attention vector should remain mostly unchanged under input perturbation, indicating stable explanations; (2) The prediction distribution should be robust to perturbations. To achieve this, we propose a new method called Denoised Diffusion Smoothing (DDS), which adopts randomized smoothing and diffusion-based denoising. We theoretically prove that processing ViTs directly with DDS can turn them into FViTs. We also show that Gaussian noise is nearly optimal for both $\ell_2$ and $\ell_\infty$-norm cases. Finally, we demonstrate the effectiveness of our approach through comprehensive experiments and evaluations. Specifically, we compare our FViTs with other baselines through visual interpretation and robustness accuracy under adversarial attacks. Results show that FViTs are more robust against adversarial attacks while maintaining the explainability of attention, indicating higher faithfulness.

摘要: 视觉转换器（ViTs）在各种视觉任务中实现了最先进的性能。成功背后的一个原因在于他们能够为神经结构的行为提供合理的先天解释。然而，ViTs存在解释忠实性的问题，因为它们的焦点对于对抗性攻击是脆弱的，并且可以很容易地通过输入图像上的轻微扰动而改变。在本文中，我们提出了一个严格的方法来减轻这些问题，通过引入忠实的ViTs（FViTs）。简而言之，FViT应该具有以下两个性质：（1）其自注意向量的前k$索引在输入扰动下应该保持基本不变，表明稳定的解释;（2）预测分布应该对扰动具有鲁棒性。为了实现这一点，我们提出了一种新的方法称为去噪扩散平滑（DDS），它采用随机平滑和基于扩散的去噪。我们从理论上证明了直接用DDS处理ViT可以将它们转化为FViT。我们还表明，高斯噪声是近最佳的$\ell_2$和$\ell_\infty$-范数的情况下。最后，我们证明了我们的方法的有效性，通过全面的实验和评估。具体来说，我们通过视觉解释和对抗性攻击下的鲁棒性准确性将我们的FViT与其他基线进行比较。结果表明，FViTs对对抗性攻击更鲁棒，同时保持注意力的可解释性，表明更高的忠诚度。



## **11. On the Adversarial Robustness of Graph Contrastive Learning Methods**

关于图对比学习方法的对抗稳健性 cs.LG

Accepted at NeurIPS 2023 New Frontiers in Graph Learning Workshop  (NeurIPS GLFrontiers 2023)

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17853v1) [paper-pdf](http://arxiv.org/pdf/2311.17853v1)

**Authors**: Filippo Guerranti, Zinuo Yi, Anna Starovoit, Rafiq Kamel, Simon Geisler, Stephan Günnemann

**Abstract**: Contrastive learning (CL) has emerged as a powerful framework for learning representations of images and text in a self-supervised manner while enhancing model robustness against adversarial attacks. More recently, researchers have extended the principles of contrastive learning to graph-structured data, giving birth to the field of graph contrastive learning (GCL). However, whether GCL methods can deliver the same advantages in adversarial robustness as their counterparts in the image and text domains remains an open question. In this paper, we introduce a comprehensive robustness evaluation protocol tailored to assess the robustness of GCL models. We subject these models to adaptive adversarial attacks targeting the graph structure, specifically in the evasion scenario. We evaluate node and graph classification tasks using diverse real-world datasets and attack strategies. With our work, we aim to offer insights into the robustness of GCL methods and hope to open avenues for potential future research directions.

摘要: 对比学习（CL）已经成为一个强大的框架，用于以自我监督的方式学习图像和文本的表示，同时增强模型对对抗性攻击的鲁棒性。最近，研究人员将对比学习的原理扩展到图结构数据，从而诞生了图对比学习（GCL）领域。然而，GCL方法是否可以在对抗鲁棒性方面提供与图像和文本领域中的对应方法相同的优势仍然是一个悬而未决的问题。在本文中，我们介绍了一个全面的鲁棒性评估协议，专门评估GCL模型的鲁棒性。我们对这些模型进行针对图结构的自适应对抗攻击，特别是在逃避场景中。我们使用不同的真实世界数据集和攻击策略来评估节点和图分类任务。通过我们的工作，我们的目标是深入了解GCL方法的鲁棒性，并希望为未来潜在的研究方向开辟道路。



## **12. SenTest: Evaluating Robustness of Sentence Encoders**

SenTest：评估句子编码器的健壮性 cs.CL

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17722v1) [paper-pdf](http://arxiv.org/pdf/2311.17722v1)

**Authors**: Tanmay Chavan, Shantanu Patankar, Aditya Kane, Omkar Gokhale, Geetanjali Kale, Raviraj Joshi

**Abstract**: Contrastive learning has proven to be an effective method for pre-training models using weakly labeled data in the vision domain. Sentence transformers are the NLP counterparts to this architecture, and have been growing in popularity due to their rich and effective sentence representations. Having effective sentence representations is paramount in multiple tasks, such as information retrieval, retrieval augmented generation (RAG), and sentence comparison. Keeping in mind the deployability factor of transformers, evaluating the robustness of sentence transformers is of utmost importance. This work focuses on evaluating the robustness of the sentence encoders. We employ several adversarial attacks to evaluate its robustness. This system uses character-level attacks in the form of random character substitution, word-level attacks in the form of synonym replacement, and sentence-level attacks in the form of intra-sentence word order shuffling. The results of the experiments strongly undermine the robustness of sentence encoders. The models produce significantly different predictions as well as embeddings on perturbed datasets. The accuracy of the models can fall up to 15 percent on perturbed datasets as compared to unperturbed datasets. Furthermore, the experiments demonstrate that these embeddings does capture the semantic and syntactic structure (sentence order) of sentences. However, existing supervised classification strategies fail to leverage this information, and merely function as n-gram detectors.

摘要: 对比学习已被证明是在视觉领域使用弱标记数据进行预训练模型的一种有效方法。句子转换器是这种体系结构的NLP对应物，由于其丰富而有效的句子表示形式而越来越受欢迎。在信息检索、检索增强生成(RAG)和句子比较等多项任务中，拥有有效的句子表征是至关重要的。考虑到转换器的可部署性因素，评估语句转换器的健壮性至关重要。这项工作的重点是评估句子编码器的健壮性。我们使用了几种对抗性攻击来评估它的健壮性。该系统使用了以随机字符替换形式的字符级攻击、以同义词替换形式的词级攻击和以句内语序洗牌形式的句子级攻击。实验结果严重削弱了句子编码器的健壮性。这些模型产生了显著不同的预测以及对扰动数据集的嵌入。与未受干扰的数据集相比，该模型在扰动数据集上的准确率最高可下降15%。此外，实验表明，这些嵌入确实捕捉到了句子的语义和句法结构(句序)。然而，现有的监督分类策略不能利用这些信息，而仅仅起到n元语法检测器的作用。



## **13. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

SmoothLLM：保护大型语言模型免受越狱攻击 cs.LG

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2310.03684v3) [paper-pdf](http://arxiv.org/pdf/2310.03684v3)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM. Our code is publicly available at the following link: https://github.com/arobey1/smooth-llm.

摘要: 尽管努力使大型语言模型(LLM)与人类价值观保持一致，但GPT、Llama、Claude和Palm等广泛使用的LLM容易受到越狱攻击，即对手欺骗目标LLM生成令人反感的内容。为了解决这一漏洞，我们提出了SmoothLLM，这是第一个旨在缓解对LLM的越狱攻击的算法。基于我们的发现，对抗性生成的提示对字符级别的变化很脆弱，我们的防御首先随机扰动给定输入提示的多个副本，然后聚合相应的预测来检测对抗性输入。SmoothLLM将许多流行的LLM的攻击成功率降低到1个百分点以下，避免了不必要的保守主义，并承认了对攻击缓解的可证明保证。此外，我们的防御使用的查询比现有攻击少得多，并且与任何LLM兼容。我们的代码可通过以下链接公开获得：https://github.com/arobey1/smooth-llm.



## **14. Natural & Adversarial Bokeh Rendering via Circle-of-Confusion Predictive Network**

基于混淆环预测网络的自然与对抗性Bokeh绘制 cs.CV

11 pages, accepted by TMM

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2111.12971v3) [paper-pdf](http://arxiv.org/pdf/2111.12971v3)

**Authors**: Yihao Huang, Felix Juefei-Xu, Qing Guo, Geguang Pu, Yang Liu

**Abstract**: Bokeh effect is a natural shallow depth-of-field phenomenon that blurs the out-of-focus part in photography. In recent years, a series of works have proposed automatic and realistic bokeh rendering methods for artistic and aesthetic purposes. They usually employ cutting-edge data-driven deep generative networks with complex training strategies and network architectures. However, these works neglect that the bokeh effect, as a real phenomenon, can inevitably affect the subsequent visual intelligent tasks like recognition, and their data-driven nature prevents them from studying the influence of bokeh-related physical parameters (i.e., depth-of-the-field) on the intelligent tasks. To fill this gap, we study a totally new problem, i.e., natural & adversarial bokeh rendering, which consists of two objectives: rendering realistic and natural bokeh and fooling the visual perception models (i.e., bokeh-based adversarial attack). To this end, beyond the pure data-driven solution, we propose a hybrid alternative by taking the respective advantages of data-driven and physical-aware methods. Specifically, we propose the circle-of-confusion predictive network (CoCNet) by taking the all-in-focus image and depth image as inputs to estimate circle-of-confusion parameters for each pixel, which are employed to render the final image through a well-known physical model of bokeh. With the hybrid solution, our method could achieve more realistic rendering results with the naive training strategy and a much lighter network.

摘要: 波克效应是一种自然的浅景深现象，它会模糊摄影中的失焦部分。近年来，出于艺术和审美的目的，一系列作品提出了自动和逼真的bokeh绘制方法。他们通常使用尖端的数据驱动的深度生成网络，具有复杂的训练策略和网络架构。然而，这些工作忽略了波克效应作为一种真实的现象，不可避免地会影响后续的视觉智能任务，如识别，其数据驱动的性质阻碍了他们研究波克相关的物理参数(即景深)对智能任务的影响。为了填补这一空白，我们研究了一个全新的问题，即自然和对抗性的bokeh绘制，它包括两个目标：渲染逼真的自然bokeh和愚弄视觉感知模型(即基于bokeh的对抗性攻击)。为此，除了纯数据驱动的解决方案之外，我们还提出了一种结合数据驱动和物理感知方法各自优势的混合替代方案。具体地说，我们提出了混淆圈预测网络(CoCNet)，它以全焦图像和深度图像作为输入来估计每个像素的混淆圈参数，并利用这些参数通过一个著名的Bokeh物理模型来呈现最终的图像。使用混合方法，我们的方法可以在简单的训练策略和更轻的网络环境下获得更逼真的渲染结果。



## **15. Query-Relevant Images Jailbreak Large Multi-Modal Models**

与查询相关的图像越狱大型多模式模型 cs.CV

Technique report

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17600v1) [paper-pdf](http://arxiv.org/pdf/2311.17600v1)

**Authors**: Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: Warning: This paper contains examples of harmful language and images, and reader discretion is recommended. The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Large Multi-Modal Models (LMMs) remains understudied. In our study, we present a novel visual prompt attack that exploits query-relevant images to jailbreak the open-source LMMs. Our method creates a composite image from one image generated by diffusion models and another that displays the text as typography, based on keywords extracted from a malicious query. We show LLMs can be easily attacked by our approach, even if the employed Large Language Models are safely aligned. To evaluate the extent of this vulnerability in open-source LMMs, we have compiled a substantial dataset encompassing 13 scenarios with a total of 5,040 text-image pairs, using our presented attack technique. Our evaluation of 12 cutting-edge LMMs using this dataset shows the vulnerability of existing multi-modal models on adversarial attacks. This finding underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source LMMs against potential malicious exploits. The resource is available at \href{this https URL}{https://github.com/isXinLiu/MM-SafetyBench}.

摘要: 警告：本文包含有害语言和图片的例子，建议读者自行决定。围绕大型语言模型(LLM)的安全问题已经得到了广泛的研究，但大型多模式模型(LMM)的安全性仍未得到充分研究。在我们的研究中，我们提出了一种新的视觉提示攻击，利用与查询相关的图像来越狱开源的LMM。我们的方法从一个由扩散模型生成的图像和另一个基于从恶意查询中提取的关键字将文本显示为排版的图像创建合成图像。我们表明，即使所使用的大型语言模型安全地对齐，LLM也可以很容易地被我们的方法攻击。为了评估这一漏洞在开源LMM中的程度，我们使用我们提出的攻击技术编制了一个包含13个场景的大量数据集，总共有5,040个文本-图像对。我们使用这个数据集对12个尖端的LMM进行了评估，表明了现有的多模式模型在对抗攻击时的脆弱性。这一发现强调了需要共同努力，加强和改进开放源码LMM的安全措施，以防范潜在的恶意利用。该资源位于\href{此HTTPS URL}{https://github.com/isXinLiu/MM-SafetyBench}.



## **16. Quantum Neural Networks under Depolarization Noise: Exploring White-Box Attacks and Defenses**

去极化噪声下的量子神经网络：白盒攻击与防御探索 quant-ph

Poster at Quantum Techniques in Machine Learning (QTML) 2023

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17458v1) [paper-pdf](http://arxiv.org/pdf/2311.17458v1)

**Authors**: David Winderl, Nicola Franco, Jeanette Miriam Lorenz

**Abstract**: Leveraging the unique properties of quantum mechanics, Quantum Machine Learning (QML) promises computational breakthroughs and enriched perspectives where traditional systems reach their boundaries. However, similarly to classical machine learning, QML is not immune to adversarial attacks. Quantum adversarial machine learning has become instrumental in highlighting the weak points of QML models when faced with adversarial crafted feature vectors. Diving deep into this domain, our exploration shines light on the interplay between depolarization noise and adversarial robustness. While previous results enhanced robustness from adversarial threats through depolarization noise, our findings paint a different picture. Interestingly, adding depolarization noise discontinued the effect of providing further robustness for a multi-class classification scenario. Consolidating our findings, we conducted experiments with a multi-class classifier adversarially trained on gate-based quantum simulators, further elucidating this unexpected behavior.

摘要: 利用量子力学的独特性质，量子机器学习（QML）有望在传统系统达到其边界的地方实现计算突破和丰富的观点。然而，与经典的机器学习类似，QML也不能免疫对抗性攻击。量子对抗机器学习在面对对抗性特征向量时，已经成为突出QML模型弱点的工具。深入研究这个领域，我们的探索揭示了去极化噪声和对抗鲁棒性之间的相互作用。虽然以前的结果通过去极化噪声增强了对抗性威胁的鲁棒性，但我们的研究结果描绘了一幅不同的画面。有趣的是，添加去极化噪声中断了为多类分类场景提供进一步鲁棒性的效果。为了巩固我们的研究结果，我们使用在基于门的量子模拟器上进行对抗训练的多类分类器进行了实验，进一步阐明了这种意想不到的行为。



## **17. Group-wise Sparse and Explainable Adversarial Attacks**

群组稀疏和可解释的对抗性攻击 cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17434v1) [paper-pdf](http://arxiv.org/pdf/2311.17434v1)

**Authors**: Shpresim Sadiku, Moritz Wagner, Sebastian Pokutta

**Abstract**: Sparse adversarial attacks fool deep neural networks (DNNs) through minimal pixel perturbations, typically regularized by the $\ell_0$ norm. Recent efforts have replaced this norm with a structural sparsity regularizer, such as the nuclear group norm, to craft group-wise sparse adversarial attacks. The resulting perturbations are thus explainable and hold significant practical relevance, shedding light on an even greater vulnerability of DNNs than previously anticipated. However, crafting such attacks poses an optimization challenge, as it involves computing norms for groups of pixels within a non-convex objective. In this paper, we tackle this challenge by presenting an algorithm that simultaneously generates group-wise sparse attacks within semantically meaningful areas of an image. In each iteration, the core operation of our algorithm involves the optimization of a quasinorm adversarial loss. This optimization is achieved by employing the $1/2$-quasinorm proximal operator for some iterations, a method tailored for nonconvex programming. Subsequently, the algorithm transitions to a projected Nesterov's accelerated gradient descent with $2$-norm regularization applied to perturbation magnitudes. We rigorously evaluate the efficacy of our novel attack in both targeted and non-targeted attack scenarios, on CIFAR-10 and ImageNet datasets. When compared to state-of-the-art methods, our attack consistently results in a remarkable increase in group-wise sparsity, e.g., an increase of $48.12\%$ on CIFAR-10 and $40.78\%$ on ImageNet (average case, targeted attack), all while maintaining lower perturbation magnitudes. Notably, this performance is complemented by a significantly faster computation time and a $100\%$ attack success rate.

摘要: 稀疏敌意攻击通过最小的像素扰动欺骗深度神经网络(DNN)，通常由$\ell_0$范数正则化。最近的努力已经用结构稀疏性正则化规则取代了这一规范，例如核集团规范，以制定群组稀疏对抗性攻击。因此，由此产生的扰动是可以解释的，并具有重要的实际意义，揭示了DNN比之前预期的更大的脆弱性。然而，精心设计这样的攻击构成了一个优化挑战，因为它涉及到计算非凸目标内的像素组的规范。在本文中，我们通过提出一种算法来解决这一挑战，该算法可以在图像的语义有意义的区域内同时生成分组稀疏攻击。在每一次迭代中，我们算法的核心操作都涉及到对一个拟正态对抗性损失的优化。这种优化是通过使用$1/2$-拟正态逼近算子进行一些迭代实现的，这是一种为非凸规划量身定做的方法。随后，算法过渡到投影的内斯特罗夫加速梯度下降，并对摄动幅度应用$2范数正则化。我们在CIFAR-10和ImageNet数据集上严格评估了我们的新型攻击在目标攻击和非目标攻击场景中的有效性。与最先进的攻击方法相比，我们的攻击始终导致组稀疏性的显著增加，例如，在CIFAR-10上增加了48.12美元，在ImageNet(平均情况下，有针对性的攻击)上增加了40.78美元，所有这些都保持了较低的扰动幅度。值得注意的是，这一性能得到了显著更快的计算时间和100美元的攻击成功率的补充。



## **18. Enhancing Adversarial Attacks: The Similar Target Method**

加强对抗性攻击：相似靶法 cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2308.10743v3) [paper-pdf](http://arxiv.org/pdf/2308.10743v3)

**Authors**: Shuo Zhang, Ziruo Wang, Zikai Zhou, Huanran Chen

**Abstract**: Deep neural networks are vulnerable to adversarial examples, posing a threat to the models' applications and raising security concerns. An intriguing property of adversarial examples is their strong transferability. Several methods have been proposed to enhance transferability, including ensemble attacks which have demonstrated their efficacy. However, prior approaches simply average logits, probabilities, or losses for model ensembling, lacking a comprehensive analysis of how and why model ensembling significantly improves transferability. In this paper, we propose a similar targeted attack method named Similar Target~(ST). By promoting cosine similarity between the gradients of each model, our method regularizes the optimization direction to simultaneously attack all surrogate models. This strategy has been proven to enhance generalization ability. Experimental results on ImageNet validate the effectiveness of our approach in improving adversarial transferability. Our method outperforms state-of-the-art attackers on 18 discriminative classifiers and adversarially trained models.

摘要: 深度神经网络很容易受到敌意例子的攻击，这对模型的应用构成了威胁，并引发了安全担忧。对抗性例子的一个耐人寻味的特点是它们具有很强的可转移性。已经提出了几种提高可转移性的方法，包括已经证明其有效性的集合攻击。然而，以前的方法只是对模型集成的对数、概率或损失进行平均，缺乏对模型集成如何以及为什么显著提高可转移性的全面分析。本文提出了一种类似的目标攻击方法--相似目标~(ST)。通过提高每个模型梯度之间的余弦相似度，我们的方法将优化方向正则化以同时攻击所有代理模型。实践证明，该策略提高了泛化能力。在ImageNet上的实验结果验证了该方法在提高对手可转移性方面的有效性。我们的方法在18个区分分类器和对抗性训练的模型上优于最先进的攻击者。



## **19. RADAP: A Robust and Adaptive Defense Against Diverse Adversarial Patches on Face Recognition**

RADAP：一种鲁棒的自适应防御人脸识别中的不同对抗补丁 cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17339v1) [paper-pdf](http://arxiv.org/pdf/2311.17339v1)

**Authors**: Xiaoliang Liu, Furao Shen, Jian Zhao, Changhai Nie

**Abstract**: Face recognition (FR) systems powered by deep learning have become widely used in various applications. However, they are vulnerable to adversarial attacks, especially those based on local adversarial patches that can be physically applied to real-world objects. In this paper, we propose RADAP, a robust and adaptive defense mechanism against diverse adversarial patches in both closed-set and open-set FR systems. RADAP employs innovative techniques, such as FCutout and F-patch, which use Fourier space sampling masks to improve the occlusion robustness of the FR model and the performance of the patch segmenter. Moreover, we introduce an edge-aware binary cross-entropy (EBCE) loss function to enhance the accuracy of patch detection. We also present the split and fill (SAF) strategy, which is designed to counter the vulnerability of the patch segmenter to complete white-box adaptive attacks. We conduct comprehensive experiments to validate the effectiveness of RADAP, which shows significant improvements in defense performance against various adversarial patches, while maintaining clean accuracy higher than that of the undefended Vanilla model.

摘要: 基于深度学习的人脸识别(FR)系统已广泛应用于各种应用领域。然而，它们很容易受到对抗性攻击，特别是那些基于可以物理应用于真实世界对象的本地对抗性补丁的攻击。在本文中，我们提出了一种在闭集和开集FR系统中针对不同敌意补丁的健壮和自适应防御机制RADAP。RADAP采用了创新的技术，如FCutout和F-Patch，它们使用傅立叶空间采样掩码来提高FR模型的遮挡稳健性和补片分割器的性能。此外，我们还引入了边缘感知的二进制交叉熵(EBCE)损失函数来提高斑块检测的准确性。针对补丁分割器在完成白盒自适应攻击时的脆弱性，提出了拆分填充(SAF)策略。我们进行了全面的实验来验证RADAP的有效性，它在对各种敌意补丁的防御性能上有了显著的提高，同时保持了比无防御的Vanilla模型更高的准确率。



## **20. NeRFTAP: Enhancing Transferability of Adversarial Patches on Face Recognition using Neural Radiance Fields**

NeRFTAP：利用神经辐射场增强人脸识别中敌方补丁的可转移性 cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17332v1) [paper-pdf](http://arxiv.org/pdf/2311.17332v1)

**Authors**: Xiaoliang Liu, Furao Shen, Feng Han, Jian Zhao, Changhai Nie

**Abstract**: Face recognition (FR) technology plays a crucial role in various applications, but its vulnerability to adversarial attacks poses significant security concerns. Existing research primarily focuses on transferability to different FR models, overlooking the direct transferability to victim's face images, which is a practical threat in real-world scenarios. In this study, we propose a novel adversarial attack method that considers both the transferability to the FR model and the victim's face image, called NeRFTAP. Leveraging NeRF-based 3D-GAN, we generate new view face images for the source and target subjects to enhance transferability of adversarial patches. We introduce a style consistency loss to ensure the visual similarity between the adversarial UV map and the target UV map under a 0-1 mask, enhancing the effectiveness and naturalness of the generated adversarial face images. Extensive experiments and evaluations on various FR models demonstrate the superiority of our approach over existing attack techniques. Our work provides valuable insights for enhancing the robustness of FR systems in practical adversarial settings.

摘要: 人脸识别(FR)技术在各种应用中扮演着至关重要的角色，但其对对手攻击的脆弱性引发了重大的安全问题。现有的研究主要集中在对不同FR模型的可转移性，而忽略了对受害者面部图像的直接可转移性，这在现实世界场景中是一种实际威胁。在这项研究中，我们提出了一种新的对抗性攻击方法，它同时考虑了对FR模型的可转换性和受害者的面部图像，称为NeRFTAP。利用基于神经网络的3D-GAN算法，为源对象和目标对象生成新的视角人脸图像，以增强对抗性补丁的可转移性。通过引入风格一致性损失，在0-1掩码下保证了敌方UV图与目标UV图的视觉相似性，增强了生成的敌方人脸图像的有效性和自然性。在各种FR模型上的广泛实验和评估证明了该方法相对于现有攻击技术的优越性。我们的工作为增强FR系统在实际对抗环境中的稳健性提供了有价值的见解。



## **21. Content-based Unrestricted Adversarial Attack**

基于内容的无限制对抗性攻击 cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2305.10665v2) [paper-pdf](http://arxiv.org/pdf/2305.10665v2)

**Authors**: Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, Wenqiang Zhang

**Abstract**: Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic, demonstrating their ability to deceive human perception and deep neural networks with stealth and success. However, current works usually sacrifice unrestricted degrees and subjectively select some image content to guarantee the photorealism of unrestricted adversarial examples, which limits its attack performance. To ensure the photorealism of adversarial examples and boost attack performance, we propose a novel unrestricted attack framework called Content-based Unrestricted Adversarial Attack. By leveraging a low-dimensional manifold that represents natural images, we map the images onto the manifold and optimize them along its adversarial direction. Therefore, within this framework, we implement Adversarial Content Attack based on Stable Diffusion and can generate high transferable unrestricted adversarial examples with various adversarial contents. Extensive experimentation and visualization demonstrate the efficacy of ACA, particularly in surpassing state-of-the-art attacks by an average of 13.3-50.4% and 16.8-48.0% in normally trained models and defense methods, respectively.

摘要: 不受限制的对抗性攻击通常会操纵图像的语义内容(例如，颜色或纹理)，以创建既有效又逼真的对抗性示例，展示它们以隐蔽和成功的方式欺骗人类感知和深层神经网络的能力。然而，目前的作品往往牺牲不受限制的程度，主观地选择一些图像内容来保证不受限制的对抗性例子的照片真实感，这限制了其攻击性能。为了保证对抗性实例的真实感，提高攻击性能，我们提出了一种新的无限制攻击框架，称为基于内容的无限对抗性攻击。通过利用表示自然图像的低维流形，我们将图像映射到流形上，并沿着其相反的方向进行优化。因此，在该框架下，我们实现了基于稳定扩散的对抗性内容攻击，并且可以生成具有多种对抗性内容的高可转移性的无限制对抗性实例。广泛的实验和可视化证明了蚁群算法的有效性，特别是在正常训练的模型和防御方法上，平均分别超过最先进的攻击13.3%-50.4%和16.8%-48.0%。



## **22. Advancing Attack-Resilient Scheduling of Integrated Energy Systems with Demand Response via Deep Reinforcement Learning**

基于深度强化学习的需求响应集成能源系统攻击弹性调度 eess.SY

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.17941v1) [paper-pdf](http://arxiv.org/pdf/2311.17941v1)

**Authors**: Yang Li, Wenjie Ma, Yuanzheng Li, Sen Li, Zhe Chen

**Abstract**: Optimally scheduling multi-energy flow is an effective method to utilize renewable energy sources (RES) and improve the stability and economy of integrated energy systems (IES). However, the stable demand-supply of IES faces challenges from uncertainties that arise from RES and loads, as well as the increasing impact of cyber-attacks with advanced information and communication technologies adoption. To address these challenges, this paper proposes an innovative model-free resilience scheduling method based on state-adversarial deep reinforcement learning (DRL) for integrated demand response (IDR)-enabled IES. The proposed method designs an IDR program to explore the interaction ability of electricity-gas-heat flexible loads. Additionally, a state-adversarial Markov decision process (SA-MDP) model characterizes the energy scheduling problem of IES under cyber-attack. The state-adversarial soft actor-critic (SA-SAC) algorithm is proposed to mitigate the impact of cyber-attacks on the scheduling strategy. Simulation results demonstrate that our method is capable of adequately addressing the uncertainties resulting from RES and loads, mitigating the impact of cyber-attacks on the scheduling strategy, and ensuring a stable demand supply for various energy sources. Moreover, the proposed method demonstrates resilience against cyber-attacks. Compared to the original soft actor-critic (SAC) algorithm, it achieves a 10\% improvement in economic performance under cyber-attack scenarios.

摘要: 多能流优化调度是利用可再生能源、提高综合能源系统稳定性和经济性的有效方法。然而，工业企业稳定的需求供应面临着挑战，这些挑战来自资源和负载带来的不确定性，以及采用先进信息和通信技术的网络攻击的影响越来越大。针对这些挑战，提出了一种基于状态对抗性深度强化学习(DRL)的集成需求响应(IDR)支持的IES的无模型弹性调度方法。该方法设计了一个IDR程序来研究电-气-热柔性负荷的相互作用能力。此外，状态对抗马尔可夫决策过程(SA-MDP)模型刻画了网络攻击下IES的能量调度问题。为了缓解网络攻击对调度策略的影响，提出了状态对抗性软行动者-批评者(SA-SAC)算法。仿真结果表明，该方法能够很好地处理资源和负荷带来的不确定性，减轻网络攻击对调度策略的影响，保证各种能源的稳定需求。此外，该方法还表现出了对网络攻击的恢复能力。与原有的软演员-批评者(SAC)算法相比，该算法在网络攻击场景下的经济性能提高了10%。



## **23. Scalable Extraction of Training Data from (Production) Language Models**

从(产生式)语言模型中可伸缩地提取训练数据 cs.LG

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.17035v1) [paper-pdf](http://arxiv.org/pdf/2311.17035v1)

**Authors**: Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramèr, Katherine Lee

**Abstract**: This paper studies extractable memorization: training data that an adversary can efficiently extract by querying a machine learning model without prior knowledge of the training dataset. We show an adversary can extract gigabytes of training data from open-source language models like Pythia or GPT-Neo, semi-open models like LLaMA or Falcon, and closed models like ChatGPT. Existing techniques from the literature suffice to attack unaligned models; in order to attack the aligned ChatGPT, we develop a new divergence attack that causes the model to diverge from its chatbot-style generations and emit training data at a rate 150x higher than when behaving properly. Our methods show practical attacks can recover far more data than previously thought, and reveal that current alignment techniques do not eliminate memorization.

摘要: 本文研究了可提取记忆：对手可以通过查询机器学习模型有效地提取训练数据，而不需要事先知道训练数据集。我们展示了对手可以从开源语言模型(如Pythia或GPT-Neo)、半开放模型(如骆驼或猎鹰)以及封闭式模型(如ChatGPT)中提取千兆字节的训练数据。现有的文献技术足以攻击未对齐的模型；为了攻击对齐的ChatGPT，我们开发了一种新的发散攻击，该攻击导致模型偏离其聊天机器人风格的代，并以比正常行为高150倍的速率发出训练数据。我们的方法表明，实际的攻击可以恢复比之前认为的更多的数据，并揭示了当前的比对技术并没有消除记忆。



## **24. Breaking Boundaries: Balancing Performance and Robustness in Deep Wireless Traffic Forecasting**

打破界限：深度无线流量预测中的性能和稳健性平衡 cs.LG

Accepted for presentation at the ARTMAN workshop, part of the ACM  Conference on Computer and Communications Security (CCS), 2023

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.09790v3) [paper-pdf](http://arxiv.org/pdf/2311.09790v3)

**Authors**: Romain Ilbert, Thai V. Hoang, Zonghua Zhang, Themis Palpanas

**Abstract**: Balancing the trade-off between accuracy and robustness is a long-standing challenge in time series forecasting. While most of existing robust algorithms have achieved certain suboptimal performance on clean data, sustaining the same performance level in the presence of data perturbations remains extremely hard. In this paper, we study a wide array of perturbation scenarios and propose novel defense mechanisms against adversarial attacks using real-world telecom data. We compare our strategy against two existing adversarial training algorithms under a range of maximal allowed perturbations, defined using $\ell_{\infty}$-norm, $\in [0.1,0.4]$. Our findings reveal that our hybrid strategy, which is composed of a classifier to detect adversarial examples, a denoiser to eliminate noise from the perturbed data samples, and a standard forecaster, achieves the best performance on both clean and perturbed data. Our optimal model can retain up to $92.02\%$ the performance of the original forecasting model in terms of Mean Squared Error (MSE) on clean data, while being more robust than the standard adversarially trained models on perturbed data. Its MSE is 2.71$\times$ and 2.51$\times$ lower than those of comparing methods on normal and perturbed data, respectively. In addition, the components of our models can be trained in parallel, resulting in better computational efficiency. Our results indicate that we can optimally balance the trade-off between the performance and robustness of forecasting models by improving the classifier and denoiser, even in the presence of sophisticated and destructive poisoning attacks.

摘要: 平衡准确性和鲁棒性之间的权衡是时间序列预测中的一个长期挑战。虽然大多数现有的鲁棒算法已经在干净数据上实现了某些次优性能，但在存在数据扰动的情况下保持相同的性能水平仍然非常困难。在本文中，我们研究了各种各样的扰动场景，并使用真实世界的电信数据提出了对抗性攻击的新防御机制。我们将我们的策略与两个现有的对抗性训练算法在最大允许扰动范围内进行比较，使用$\ell_{\infty}$-norm，$\in [0.1，0.4]$定义。我们的研究结果表明，我们的混合策略，它是由一个分类器来检测对抗性的例子，一个去噪器，以消除干扰的数据样本中的噪声，和一个标准的预测器，实现了最好的性能在干净和扰动的数据。我们的最佳模型可以保留高达92.02\%$的原始预测模型的性能方面的均方误差（MSE）干净的数据，而更强大的干扰数据比标准的对抗训练模型。在正态和扰动数据下，其均方误差分别比比较方法低2.71倍和2.51倍。此外，我们模型的组件可以并行训练，从而提高计算效率。我们的研究结果表明，我们可以通过改进分类器和去噪器来最佳地平衡预测模型的性能和鲁棒性之间的权衡，即使在存在复杂和破坏性的中毒攻击的情况下。



## **25. Vulnerability Analysis of Transformer-based Optical Character Recognition to Adversarial Attacks**

基于变压器的光学字符识别对敌方攻击的脆弱性分析 cs.CV

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.17128v1) [paper-pdf](http://arxiv.org/pdf/2311.17128v1)

**Authors**: Lucas Beerens, Desmond J. Higham

**Abstract**: Recent advancements in Optical Character Recognition (OCR) have been driven by transformer-based models. OCR systems are critical in numerous high-stakes domains, yet their vulnerability to adversarial attack remains largely uncharted territory, raising concerns about security and compliance with emerging AI regulations. In this work we present a novel framework to assess the resilience of Transformer-based OCR (TrOCR) models. We develop and assess algorithms for both targeted and untargeted attacks. For the untargeted case, we measure the Character Error Rate (CER), while for the targeted case we use the success ratio. We find that TrOCR is highly vulnerable to untargeted attacks and somewhat less vulnerable to targeted attacks. On a benchmark handwriting data set, untargeted attacks can cause a CER of more than 1 without being noticeable to the eye. With a similar perturbation size, targeted attacks can lead to success rates of around $25\%$ -- here we attacked single tokens, requiring TrOCR to output the tenth most likely token from a large vocabulary.

摘要: 光学字符识别(OCR)的最新进展是由基于变压器的模型驱动的。OCR系统在许多高风险领域至关重要，但它们对对手攻击的脆弱性在很大程度上仍然是未知领域，这引发了人们对安全和遵守新兴人工智能法规的担忧。在这项工作中，我们提出了一个新的框架来评估基于变压器的OCR(TrOCR)模型的弹性。我们开发和评估针对目标攻击和非目标攻击的算法。对于非目标情况，我们测量字符错误率(CER)，而对于目标情况，我们使用成功率。我们发现TrOCR非常容易受到非目标攻击，而不太容易受到目标攻击。在基准手写数据集上，无目标攻击可以导致CER大于1，而不会引起肉眼可见的CER。使用类似的扰动大小，有针对性的攻击可以导致大约$25\$的成功率--在这里我们攻击单个令牌，要求TrOCR从大量词汇表中输出最有可能的第十个令牌。



## **26. Generation of Games for Opponent Model Differentiation**

竞争对手模型差异化博弈的生成 cs.AI

4 pages

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16781v1) [paper-pdf](http://arxiv.org/pdf/2311.16781v1)

**Authors**: David Milec, Viliam Lisý, Christopher Kiekintveld

**Abstract**: Protecting against adversarial attacks is a common multiagent problem. Attackers in the real world are predominantly human actors, and the protection methods often incorporate opponent models to improve the performance when facing humans. Previous results show that modeling human behavior can significantly improve the performance of the algorithms. However, modeling humans correctly is a complex problem, and the models are often simplified and assume humans make mistakes according to some distribution or train parameters for the whole population from which they sample. In this work, we use data gathered by psychologists who identified personality types that increase the likelihood of performing malicious acts. However, in the previous work, the tests on a handmade game could not show strategic differences between the models. We created a novel model that links its parameters to psychological traits. We optimized over parametrized games and created games in which the differences are profound. Our work can help with automatic game generation when we need a game in which some models will behave differently and to identify situations in which the models do not align.

摘要: 防御敌意攻击是一个常见的多智能体问题。现实世界中的攻击者主要是人类演员，保护方法通常会结合对手模型来提高面对人类时的性能。以往的研究结果表明，对人的行为进行建模可以显著提高算法的性能。然而，正确地为人类建模是一个复杂的问题，模型往往被简化，并假设人类根据样本所在总体的某些分布或训练参数出错。在这项工作中，我们使用了心理学家收集的数据，他们确定了增加实施恶意行为可能性的个性类型。然而，在之前的工作中，对手工游戏的测试无法显示模型之间的战略差异。我们创建了一个新的模型，将其参数与心理特征联系起来。我们对参数化游戏进行了优化，并创建了差异巨大的游戏。当我们需要一个游戏时，我们的工作可以帮助自动生成游戏，在其中一些模型会有不同的行为，并识别模型不一致的情况。



## **27. Cooperative Abnormal Node Detection with Adversary Resistance: A Probabilistic Approach**

对抗对手的协作式异常节点检测：一种概率方法 eess.SY

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16661v1) [paper-pdf](http://arxiv.org/pdf/2311.16661v1)

**Authors**: Yingying Huangfu, Tian Bai

**Abstract**: This paper presents a novel probabilistic detection scheme called Cooperative Statistical Detection (CSD) for abnormal node detection while defending against adversarial attacks in cluster-tree networks. The CSD performs a two-phase process: 1) designing a likelihood ratio test (LRT) for a non-root node at its children from the perspective of packet loss; 2) making an overall decision at the root node based on the aggregated detection data of the nodes over tree branches. In most adversarial scenarios, malicious children knowing the detection policy can generate falsified data to protect the abnormal parent from being detected or frame its normal parent as an anomalous node. To resolve this issue, a modified Z-score-based falsification-resistant mechanism is presented in the CSD to remove untrustworthy information. Through theoretical analysis, we show that the LRT-based method achieves perfect detection, i.e., both the false alarm and missed detection probabilities decay exponentially to zero. Furthermore, the optimal removal threshold of the modified Z-score method is derived for falsifications with uncertain strategies and guarantees perfect detection of the CSD. As our simulation results show, the CSD approach is robust to falsifications and can rapidly reach $99\%$ detection accuracy, even in existing adversarial scenarios, which outperforms state-of-the-art technology.

摘要: 本文提出了一种新的概率检测方案，称为合作统计检测（CSD）的异常节点检测，同时抵御敌对攻击的簇树网络。CSD执行两阶段过程：1）从分组丢失的角度设计非根节点在其子节点处的似然比测试（LRT）; 2）基于树分支上的节点的聚合检测数据在根节点处做出总体决策。在大多数对抗性场景中，知道检测策略的恶意子节点可以生成伪造的数据，以保护异常父节点不被检测到，或者将其正常父节点框定为异常节点。为了解决这个问题，在CSD中提出了一种改进的基于Z分数的防篡改机制，以去除不可信的信息。通过理论分析，我们表明，基于LRT的方法实现了完美的检测，即，虚警概率和漏检概率都指数衰减到零。此外，修改后的Z分数方法的最佳去除阈值推导出不确定策略的伪造，并保证完美的检测CSD。正如我们的模拟结果表明，CSD方法是强大的伪造，可以迅速达到99\%$的检测精度，即使在现有的对抗性的情况下，这优于国家的最先进的技术。



## **28. On the Role of Randomization in Adversarially Robust Classification**

论随机化在逆稳性分类中的作用 cs.LG

10 pages main paper (27 total), 2 figures in main paper. Neurips 2023

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2302.07221v3) [paper-pdf](http://arxiv.org/pdf/2302.07221v3)

**Authors**: Lucas Gnecco-Heredia, Yann Chevaleyre, Benjamin Negrevergne, Laurent Meunier, Muni Sreenivas Pydi

**Abstract**: Deep neural networks are known to be vulnerable to small adversarial perturbations in test data. To defend against adversarial attacks, probabilistic classifiers have been proposed as an alternative to deterministic ones. However, literature has conflicting findings on the effectiveness of probabilistic classifiers in comparison to deterministic ones. In this paper, we clarify the role of randomization in building adversarially robust classifiers. Given a base hypothesis set of deterministic classifiers, we show the conditions under which a randomized ensemble outperforms the hypothesis set in adversarial risk, extending previous results. Additionally, we show that for any probabilistic binary classifier (including randomized ensembles), there exists a deterministic classifier that outperforms it. Finally, we give an explicit description of the deterministic hypothesis set that contains such a deterministic classifier for many types of commonly used probabilistic classifiers, i.e. randomized ensembles and parametric/input noise injection.

摘要: 众所周知，深度神经网络在测试数据中容易受到微小的对抗性扰动。为了防御敌意攻击，人们提出了概率分类器作为确定性分类器的替代方案。然而，与确定性分类器相比，文献对概率分类器的有效性有相互矛盾的发现。在这篇文章中，我们阐明了随机化在构建对抗性稳健分类器中的作用。在给定确定性分类器的基本假设集的情况下，我们证明了随机化集成在对抗风险方面优于假设集的条件，扩展了先前的结果。此外，我们还证明了对于任何概率二进制分类器(包括随机集成)，都存在一个性能优于它的确定性分类器。最后，我们对于许多常用的概率分类器，即随机集成和参数/输入噪声注入，给出了包含这种确定性分类器的确定性假设集的显式描述。



## **29. Efficient Key-Based Adversarial Defense for ImageNet by Using Pre-trained Model**

基于预训练模型的高效基于密钥的ImageNet对抗防御 cs.CV

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16577v1) [paper-pdf](http://arxiv.org/pdf/2311.16577v1)

**Authors**: AprilPyone MaungMaung, Isao Echizen, Hitoshi Kiya

**Abstract**: In this paper, we propose key-based defense model proliferation by leveraging pre-trained models and utilizing recent efficient fine-tuning techniques on ImageNet-1k classification. First, we stress that deploying key-based models on edge devices is feasible with the latest model deployment advancements, such as Apple CoreML, although the mainstream enterprise edge artificial intelligence (Edge AI) has been focused on the Cloud. Then, we point out that the previous key-based defense on on-device image classification is impractical for two reasons: (1) training many classifiers from scratch is not feasible, and (2) key-based defenses still need to be thoroughly tested on large datasets like ImageNet. To this end, we propose to leverage pre-trained models and utilize efficient fine-tuning techniques to proliferate key-based models even on limited computing resources. Experiments were carried out on the ImageNet-1k dataset using adaptive and non-adaptive attacks. The results show that our proposed fine-tuned key-based models achieve a superior classification accuracy (more than 10% increase) compared to the previous key-based models on classifying clean and adversarial examples.

摘要: 在本文中，我们通过利用预先训练的模型和利用最新的高效微调技术对ImageNet-1k分类进行微调，提出了基于密钥的防御模型扩散。首先，我们强调，尽管主流企业边缘人工智能(Edge AI)一直专注于云，但随着Apple CoreML等最新模型部署的进步，在边缘设备上部署基于密钥的模型是可行的。然后，我们指出基于密钥的防御在设备上的图像分类是不现实的，原因有两个：(1)从头开始训练许多分类器是不可行的；(2)基于密钥的防御仍然需要在像ImageNet这样的大型数据集上进行彻底的测试。为此，我们建议利用预先训练的模型并利用有效的微调技术来繁殖基于关键字的模型，即使在有限的计算资源上也是如此。使用自适应和非自适应攻击在ImageNet-1k数据集上进行了实验。实验结果表明，与已有的基于关键字的分类模型相比，本文提出的改进的基于关键字的模型具有更高的分类正确率(提高了10%以上)。



## **30. Adversarial Doodles: Interpretable and Human-drawable Attacks Provide Describable Insights**

对抗性涂鸦：可解释和可人类绘制的攻击提供可描述的洞察力 cs.CV

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.15994v2) [paper-pdf](http://arxiv.org/pdf/2311.15994v2)

**Authors**: Ryoya Nara, Yusuke Matsui

**Abstract**: DNN-based image classification models are susceptible to adversarial attacks. Most previous adversarial attacks do not focus on the interpretability of the generated adversarial examples, and we cannot gain insights into the mechanism of the target classifier from the attacks. Therefore, we propose Adversarial Doodles, which have interpretable shapes. We optimize black b\'ezier curves to fool the target classifier by overlaying them onto the input image. By introducing random perspective transformation and regularizing the doodled area, we obtain compact attacks that cause misclassification even when humans replicate them by hand. Adversarial doodles provide describable and intriguing insights into the relationship between our attacks and the classifier's output. We utilize adversarial doodles and discover the bias inherent in the target classifier, such as "We add two strokes on its head, a triangle onto its body, and two lines inside the triangle on a bird image. Then, the classifier misclassifies the image as a butterfly."

摘要: 基于DNN的图像分类模型容易受到敌意攻击。以往的对抗性攻击大多不关注生成的对抗性实例的可解释性，无法从攻击中深入了解目标分类器的作用机制。因此，我们提出了对抗性涂鸦，它具有可解释的形状。我们通过将黑色贝塞尔曲线叠加到输入图像上来优化目标分类器。通过引入随机透视变换和规则化涂鸦区域，我们获得了即使在人类手动复制它们时也会导致错误分类的紧凑攻击。对抗性涂鸦为我们的攻击和分类器输出之间的关系提供了可描述和有趣的见解。我们利用对抗性涂鸦发现目标分类器固有的偏差，例如，我们在鸟的图像上添加两个笔画，在它的身体上添加一个三角形，并在三角形内添加两条线。然后，分类器将图像错误分类为一只蝴蝶。



## **31. Threshold Breaker: Can Counter-Based RowHammer Prevention Mechanisms Truly Safeguard DRAM?**

打破门槛：基于计数器的行锤预防机制真的能保护DRAM吗？ cs.AR

7 pages, 6 figures

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16460v1) [paper-pdf](http://arxiv.org/pdf/2311.16460v1)

**Authors**: Ranyang Zhou, Jacqueline Liu, Sabbir Ahmed, Nakul Kochar, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: This paper challenges the existing victim-focused counter-based RowHammer detection mechanisms by experimentally demonstrating a novel multi-sided fault injection attack technique called Threshold Breaker. This mechanism can effectively bypass the most advanced counter-based defense mechanisms by soft-attacking the rows at a farther physical distance from the target rows. While no prior work has demonstrated the effect of such an attack, our work closes this gap by systematically testing 128 real commercial DDR4 DRAM products and reveals that the Threshold Breaker affects various chips from major DRAM manufacturers. As a case study, we compare the performance efficiency between our mechanism and a well-known double-sided attack by performing adversarial weight attacks on a modern Deep Neural Network (DNN). The results demonstrate that the Threshold Breaker can deliberately deplete the intelligence of the targeted DNN system while DRAM is fully protected.

摘要: 通过实验演示了一种新的多边故障注入攻击技术--门限断路器，从而挑战了现有的以受害者为中心的反RowHammer检测机制。该机制通过在距离目标行较远的物理距离处对行进行软攻击，可以有效地绕过最先进的基于反击的防御机制。虽然以前的工作没有证明这种攻击的影响，但我们的工作通过系统地测试128个真实的商业DDR4 DRAM产品来缩小这一差距，并揭示了Threshold Breaker会影响主要DRAM制造商的各种芯片。作为一个实例，我们通过对现代深度神经网络(DNN)进行对抗性加权攻击，比较了该机制与著名的双边攻击的性能效率。结果表明，在DRAM受到完全保护的情况下，门限断路器可以故意耗尽目标DNN系统的智能。



## **32. Rethinking Mixup for Improving the Adversarial Transferability**

对提高对抗性转移性的混搭的再思考 cs.CV

13 pages, 8 figures, 4 tables

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.17087v1) [paper-pdf](http://arxiv.org/pdf/2311.17087v1)

**Authors**: Xiaosen Wang, Zeyuan Yin

**Abstract**: Mixup augmentation has been widely integrated to generate adversarial examples with superior adversarial transferability when immigrating from a surrogate model to other models. However, the underlying mechanism influencing the mixup's effect on transferability remains unexplored. In this work, we posit that the adversarial examples located at the convergence of decision boundaries across various categories exhibit better transferability and identify that Admix tends to steer the adversarial examples towards such regions. However, we find the constraint on the added image in Admix decays its capability, resulting in limited transferability. To address such an issue, we propose a new input transformation-based attack called Mixing the Image but Separating the gradienT (MIST). Specifically, MIST randomly mixes the input image with a randomly shifted image and separates the gradient of each loss item for each mixed image. To counteract the imprecise gradient, MIST calculates the gradient on several mixed images for each input sample. Extensive experimental results on the ImageNet dataset demonstrate that MIST outperforms existing SOTA input transformation-based attacks with a clear margin on both Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) w/wo defense mechanisms, supporting MIST's high effectiveness and generality.

摘要: 当从代理模型迁移到其他模型时，混合增强已经被广泛地集成以生成具有优越的对抗性可转移性的对抗性实例。然而，影响混合对可转移性影响的潜在机制尚不清楚。在这项工作中，我们假设位于不同类别决策边界收敛处的对抗性例子表现出更好的可转移性，并发现AdMix倾向于将对抗性例子引向这些区域。然而，我们发现AdMix中对添加图像的限制降低了其性能，导致可转移性有限。为了解决这一问题，我们提出了一种新的基于输入变换的攻击方法，称为混合图像但分离梯度(MIST)。具体地，MIST将输入图像与随机移位的图像随机混合，并为每个混合图像分离每个损失项的梯度。为了抵消不精确的渐变，MIST在每个输入样本的几个混合图像上计算渐变。在ImageNet数据集上的大量实验结果表明，MIST在卷积神经网络(CNN)和视觉转换器(VITS)两种防御机制上的表现优于现有的基于SOTA输入变换的攻击，支持MIST的高效性和通用性。



## **33. Certifying LLM Safety against Adversarial Prompting**

针对敌意提示认证LLM安全 cs.CL

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2309.02705v2) [paper-pdf](http://arxiv.org/pdf/2309.02705v2)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial attacks, which add maliciously designed token sequences to a harmful prompt to bypass the model's safety guards. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We defend against three attack modes: i) adversarial suffix, which appends an adversarial sequence at the end of the prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. For example, against adversarial suffixes of length 20, it certifiably detects 92% of harmful prompts and labels 94% of safe prompts correctly using the open-source language model Llama 2 as the safety filter. We further improve the filter's performance, in terms of accuracy and speed, by replacing Llama 2 with a DistilBERT safety classifier fine-tuned on safe and harmful prompts. Additionally, we propose two efficient empirical defenses: i) RandEC, a randomized version of erase-and-check that evaluates the safety filter on a small subset of the erased subsequences, and ii) GradEC, a gradient-based version that optimizes the erased tokens to remove the adversarial sequence. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.

摘要: 发布供公众使用的大型语言模型（LLM）包含了护栏，以确保其输出是安全的，通常被称为“模型对齐”。“一个对齐的语言模型应该拒绝用户制作有害内容的请求。然而，这样的安全措施容易受到对抗性攻击，这些攻击将恶意设计的令牌序列添加到有害的提示中，以绕过模型的安全防护。在这项工作中，我们引入擦除和检查，第一个框架，以抵御对抗性提示与可验证的安全保证。我们防御三种攻击模式：i）对抗后缀，它在提示符的末尾附加一个对抗序列; ii）对抗插入，其中对抗序列被插入到提示符中间的任何位置; iii）对抗注入，其中对抗令牌被插入到提示符中的任意位置，不一定是一个连续的块。我们的实验结果表明，该程序可以获得强有力的认证安全保证有害的提示，同时保持良好的经验性能的安全提示。例如，针对长度为20的对抗性后缀，它可以使用开源语言模型Llama 2作为安全过滤器，正确检测92%的有害提示并标记94%的安全提示。我们进一步提高了过滤器的性能，在准确性和速度方面，通过用DistilBERT安全分类器替换Llama 2，对安全和有害提示进行微调。此外，我们还提出了两种有效的经验防御：i）RandEC，一种随机版本的擦除和检查，它在擦除的合法性的一个小子集上评估安全过滤器; ii）GradEC，一种基于梯度的版本，它优化擦除的令牌以删除对抗序列。我们的实验代码可以在https://github.com/aounon/certified-llm-safety上找到。



## **34. Mate! Are You Really Aware? An Explainability-Guided Testing Framework for Robustness of Malware Detectors**

伙计！你真的知道吗？一种可解释性指导的恶意软件检测器鲁棒性测试框架 cs.CR

Accepted at ESEC/FSE 2023. https://doi.org/10.1145/3611643.3616309

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2111.10085v4) [paper-pdf](http://arxiv.org/pdf/2111.10085v4)

**Authors**: Ruoxi Sun, Minhui Xue, Gareth Tyson, Tian Dong, Shaofeng Li, Shuo Wang, Haojin Zhu, Seyit Camtepe, Surya Nepal

**Abstract**: Numerous open-source and commercial malware detectors are available. However, their efficacy is threatened by new adversarial attacks, whereby malware attempts to evade detection, e.g., by performing feature-space manipulation. In this work, we propose an explainability-guided and model-agnostic testing framework for robustness of malware detectors when confronted with adversarial attacks. The framework introduces the concept of Accrued Malicious Magnitude (AMM) to identify which malware features could be manipulated to maximize the likelihood of evading detection. We then use this framework to test several state-of-the-art malware detectors' abilities to detect manipulated malware. We find that (i) commercial antivirus engines are vulnerable to AMM-guided test cases; (ii) the ability of a manipulated malware generated using one detector to evade detection by another detector (i.e., transferability) depends on the overlap of features with large AMM values between the different detectors; and (iii) AMM values effectively measure the fragility of features (i.e., capability of feature-space manipulation to flip the prediction results) and explain the robustness of malware detectors facing evasion attacks. Our findings shed light on the limitations of current malware detectors, as well as how they can be improved.

摘要: 有许多开源和商业恶意软件检测器可用。然而，它们的有效性受到新的敌意攻击的威胁，借此恶意软件试图通过例如执行特征空间操纵来逃避检测。在这项工作中，我们提出了一个可解释性指导和模型无关的测试框架，用于测试恶意软件检测器在面对敌意攻击时的健壮性。该框架引入了累积恶意量级(AMM)的概念，以确定哪些恶意软件功能可以被操纵，以最大限度地提高逃避检测的可能性。然后，我们使用这个框架来测试几个最先进的恶意软件检测器检测操纵恶意软件的能力。我们发现(I)商业反病毒引擎容易受到AMM引导的测试用例的攻击；(Ii)使用一个检测器生成的被操纵的恶意软件逃避另一个检测器的检测的能力(即可转移性)取决于不同检测器之间具有较大AMM值的特征的重叠；以及(Iii)AMM值有效地衡量了特征的脆弱性(即，对特征空间的操纵来反转预测结果的能力)，并解释了恶意软件检测器面对逃避攻击的健壮性。我们的发现揭示了当前恶意软件检测器的局限性，以及如何改进它们。



## **35. How Many Unicorns Are in This Image? A Safety Evaluation Benchmark for Vision LLMs**

这张图片中有多少只独角兽？一种视觉LLMS的安全评价基准 cs.CV

H.T., C.C., and Z.W. contribute equally. Work done during H.T. and  Z.W.'s internship at UCSC, and C.C. and Y.Z.'s internship at UNC

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.16101v1) [paper-pdf](http://arxiv.org/pdf/2311.16101v1)

**Authors**: Haoqin Tu, Chenhang Cui, Zijun Wang, Yiyang Zhou, Bingchen Zhao, Junlin Han, Wangchunshu Zhou, Huaxiu Yao, Cihang Xie

**Abstract**: This work focuses on the potential of Vision LLMs (VLLMs) in visual reasoning. Different from prior studies, we shift our focus from evaluating standard performance to introducing a comprehensive safety evaluation suite, covering both out-of-distribution (OOD) generalization and adversarial robustness. For the OOD evaluation, we present two novel VQA datasets, each with one variant, designed to test model performance under challenging conditions. In exploring adversarial robustness, we propose a straightforward attack strategy for misleading VLLMs to produce visual-unrelated responses. Moreover, we assess the efficacy of two jailbreaking strategies, targeting either the vision or language component of VLLMs. Our evaluation of 21 diverse models, ranging from open-source VLLMs to GPT-4V, yields interesting observations: 1) Current VLLMs struggle with OOD texts but not images, unless the visual information is limited; and 2) These VLLMs can be easily misled by deceiving vision encoders only, and their vision-language training often compromise safety protocols. We release this safety evaluation suite at https://github.com/UCSC-VLAA/vllm-safety-benchmark.

摘要: 本文主要研究视觉LLMS(VLLM)在视觉推理中的潜力。与以前的研究不同，我们将重点从评估标准性能转移到引入一个全面的安全评估套件，包括分布外(OOD)泛化和对手健壮性。对于面向对象的评估，我们提出了两个新的VQA数据集，每个数据集都有一个变量，旨在测试模型在具有挑战性的条件下的性能。在探索对抗健壮性的过程中，我们提出了一种直接的攻击策略，用于误导VLLM产生与视觉无关的响应。此外，我们评估了两种越狱策略的有效性，目标是VLLM的视觉或语言部分。我们对21种不同的模型进行了评估，从开源的VLLM到GPT-4V，得出了有趣的观察结果：1)当前的VLLM难以处理OOD文本而不是图像，除非视觉信息有限；2)这些VLLM很容易被欺骗的视觉编码器误导，并且它们的视觉语言培训经常危及安全协议。我们在https://github.com/UCSC-VLAA/vllm-safety-benchmark.上发布此安全评估套件



## **36. CALICO: Self-Supervised Camera-LiDAR Contrastive Pre-training for BEV Perception**

Calico：BEV感知的自我监控相机-LiDAR对比预训练 cs.CV

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2306.00349v2) [paper-pdf](http://arxiv.org/pdf/2306.00349v2)

**Authors**: Jiachen Sun, Haizhong Zheng, Qingzhao Zhang, Atul Prakash, Z. Morley Mao, Chaowei Xiao

**Abstract**: Perception is crucial in the realm of autonomous driving systems, where bird's eye view (BEV)-based architectures have recently reached state-of-the-art performance. The desirability of self-supervised representation learning stems from the expensive and laborious process of annotating 2D and 3D data. Although previous research has investigated pretraining methods for both LiDAR and camera-based 3D object detection, a unified pretraining framework for multimodal BEV perception is missing. In this study, we introduce CALICO, a novel framework that applies contrastive objectives to both LiDAR and camera backbones. Specifically, CALICO incorporates two stages: point-region contrast (PRC) and region-aware distillation (RAD). PRC better balances the region- and scene-level representation learning on the LiDAR modality and offers significant performance improvement compared to existing methods. RAD effectively achieves contrastive distillation on our self-trained teacher model. CALICO's efficacy is substantiated by extensive evaluations on 3D object detection and BEV map segmentation tasks, where it delivers significant performance improvements. Notably, CALICO outperforms the baseline method by 10.5% and 8.6% on NDS and mAP. Moreover, CALICO boosts the robustness of multimodal 3D object detection against adversarial attacks and corruption. Additionally, our framework can be tailored to different backbones and heads, positioning it as a promising approach for multimodal BEV perception.

摘要: 感知在自动驾驶系统领域至关重要，其中基于鸟瞰图（BEV）的架构最近已经达到了最先进的性能。自我监督表示学习的可取性源于注释2D和3D数据的昂贵且费力的过程。虽然以前的研究已经研究了用于LiDAR和基于相机的3D对象检测的预训练方法，但缺少用于多模态BEV感知的统一预训练框架。在这项研究中，我们介绍了CALICO，这是一种新的框架，它将对比目标应用于LiDAR和相机主干。具体而言，CALICO包括两个阶段：点区域对比（PRC）和区域感知蒸馏（RAD）。PRC更好地平衡了LiDAR模式上的区域和场景级表示学习，并与现有方法相比提供了显着的性能改进。RAD有效地实现了对我们自我培训的教师模型的对比升华。CALICO的功效通过对3D对象检测和BEV地图分割任务的广泛评估得到了证实，在这些任务中，它提供了显着的性能改进。值得注意的是，CALICO在NDS和mAP上的表现优于基线方法10.5%和8.6%。此外，CALICO增强了多模式3D对象检测对对抗性攻击和腐败的鲁棒性。此外，我们的框架可以针对不同的骨干和头部进行定制，将其定位为多模态BEV感知的有前途的方法。



## **37. AdaptGuard: Defending Against Universal Attacks for Model Adaptation**

AdaptGuard：针对模型适配的通用攻击防御 cs.CR

ICCV2023

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2303.10594v2) [paper-pdf](http://arxiv.org/pdf/2303.10594v2)

**Authors**: Lijun Sheng, Jian Liang, Ran He, Zilei Wang, Tieniu Tan

**Abstract**: Model adaptation aims at solving the domain transfer problem under the constraint of only accessing the pretrained source models. With the increasing considerations of data privacy and transmission efficiency, this paradigm has been gaining recent popularity. This paper studies the vulnerability to universal attacks transferred from the source domain during model adaptation algorithms due to the existence of malicious providers. We explore both universal adversarial perturbations and backdoor attacks as loopholes on the source side and discover that they still survive in the target models after adaptation. To address this issue, we propose a model preprocessing framework, named AdaptGuard, to improve the security of model adaptation algorithms. AdaptGuard avoids direct use of the risky source parameters through knowledge distillation and utilizes the pseudo adversarial samples under adjusted radius to enhance the robustness. AdaptGuard is a plug-and-play module that requires neither robust pretrained models nor any changes for the following model adaptation algorithms. Extensive results on three commonly used datasets and two popular adaptation methods validate that AdaptGuard can effectively defend against universal attacks and maintain clean accuracy in the target domain simultaneously. We hope this research will shed light on the safety and robustness of transfer learning. Code is available at https://github.com/TomSheng21/AdaptGuard.

摘要: 模型自适应的目的是在只访问预先训练好的源模型的约束下解决域迁移问题。随着人们对数据隐私和传输效率的越来越多的考虑，这种范式最近越来越受欢迎。本文研究了在模型自适应算法中，由于恶意提供者的存在，对源域传输的通用攻击的脆弱性。我们探索了普遍的对抗性扰动和后门攻击作为源端的漏洞，并发现它们在适应后仍然存在于目标模型中。针对这一问题，我们提出了一种模型预处理框架AdaptGuard，以提高模型自适应算法的安全性。AdaptGuard通过知识提取避免了直接使用风险源参数，并利用调整后的半径下的伪对手样本来增强鲁棒性。AdaptGuard是一个即插即用模块，它既不需要健壮的预先训练的模型，也不需要对以下模型自适应算法进行任何更改。在三个常用数据集和两个流行的自适应方法上的广泛结果验证了AdaptGuard能够有效地防御通用攻击，同时保持目标领域的干净准确性。我们希望这项研究能对迁移学习的安全性和稳健性有所帮助。代码可在https://github.com/TomSheng21/AdaptGuard.上找到



## **38. Attend Who is Weak: Enhancing Graph Condensation via Cross-Free Adversarial Training**

关注弱者：通过交叉自由对抗训练增强图形凝聚力 cs.LG

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.15772v1) [paper-pdf](http://arxiv.org/pdf/2311.15772v1)

**Authors**: Xinglin Li, Kun Wang, Hanhui Deng, Yuxuan Liang, Di Wu

**Abstract**: In this paper, we study the \textit{graph condensation} problem by compressing the large, complex graph into a concise, synthetic representation that preserves the most essential and discriminative information of structure and features. We seminally propose the concept of Shock Absorber (a type of perturbation) that enhances the robustness and stability of the original graphs against changes in an adversarial training fashion. Concretely, (I) we forcibly match the gradients between pre-selected graph neural networks (GNNs) trained on a synthetic, simplified graph and the original training graph at regularly spaced intervals. (II) Before each update synthetic graph point, a Shock Absorber serves as a gradient attacker to maximize the distance between the synthetic dataset and the original graph by selectively perturbing the parts that are underrepresented or insufficiently informative. We iteratively repeat the above two processes (I and II) in an adversarial training fashion to maintain the highly-informative context without losing correlation with the original dataset. More importantly, our shock absorber and the synthesized graph parallelly share the backward process in a free training manner. Compared to the original adversarial training, it introduces almost no additional time overhead.   We validate our framework across 8 datasets (3 graph and 5 node classification datasets) and achieve prominent results: for example, on Cora, Citeseer and Ogbn-Arxiv, we can gain nearly 1.13% to 5.03% improvements compare with SOTA models. Moreover, our algorithm adds only about 0.2% to 2.2% additional time overhead over Flicker, Citeseer and Ogbn-Arxiv. Compared to the general adversarial training, our approach improves time efficiency by nearly 4-fold.

摘要: 在这篇文章中，我们通过将大的复杂的图压缩成一个简洁的、综合的表示来研究图压缩问题，它保留了结构和特征的最本质和可区分的信息。我们半自动地提出了减震器(一种扰动)的概念，它以对抗性训练的方式增强了原始图形对变化的稳健性和稳定性。具体地说，(I)我们以规则间隔强制匹配在合成的简化图上训练的预先选择的图神经网络(GNN)和原始训练图之间的梯度。(Ii)在每次更新合成图形点之前，减震器充当梯度攻击者，通过选择性地扰动表示不足或信息不足的部分来最大化合成数据集和原始图形之间的距离。我们以对抗性训练的方式迭代重复上述两个过程(I和II)，以保持高度信息量的上下文，而不会失去与原始数据集的相关性。更重要的是，我们的减振器和合成的图形以自由训练的方式并行共享反向过程。与最初的对抗性训练相比，它几乎不会带来额外的时间开销。我们在8个数据集(3个图和5个节点分类数据集)上验证了我们的框架，并取得了显著的结果：例如，在CORA、Citeseer和Ogbn-Arxiv上，我们可以比SOTA模型获得近1.13%到5.03%的改进。此外，与Flicker、Citeseer和Ogbn-Arxiv相比，我们的算法只增加了大约0.2%到2.2%的额外时间开销。与一般的对抗性训练相比，我们的方法将时间效率提高了近4倍。



## **39. The Lipschitz-Variance-Margin Tradeoff for Enhanced Randomized Smoothing**

增强随机化平滑的Lipschitz-Variance-March权衡 cs.LG

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2309.16883v2) [paper-pdf](http://arxiv.org/pdf/2309.16883v2)

**Authors**: Blaise Delattre, Alexandre Araujo, Quentin Barthélemy, Alexandre Allauzen

**Abstract**: Real-life applications of deep neural networks are hindered by their unsteady predictions when faced with noisy inputs and adversarial attacks. The certified radius is in this context a crucial indicator of the robustness of models. However how to design an efficient classifier with a sufficient certified radius? Randomized smoothing provides a promising framework by relying on noise injection in inputs to obtain a smoothed and more robust classifier. In this paper, we first show that the variance introduced by randomized smoothing closely interacts with two other important properties of the classifier, \textit{i.e.} its Lipschitz constant and margin. More precisely, our work emphasizes the dual impact of the Lipschitz constant of the base classifier, on both the smoothed classifier and the empirical variance. Moreover, to increase the certified robust radius, we introduce a different simplex projection technique for the base classifier to leverage the variance-margin trade-off thanks to Bernstein's concentration inequality, along with an enhanced Lipschitz bound. Experimental results show a significant improvement in certified accuracy compared to current state-of-the-art methods. Our novel certification procedure allows us to use pre-trained models that are used with randomized smoothing, effectively improving the current certification radius in a zero-shot manner.

摘要: 当面对噪声输入和敌意攻击时，深度神经网络的不稳定预测阻碍了其在现实生活中的应用。在这种情况下，认证半径是模型稳健性的关键指标。然而，如何设计一个具有足够认证半径的高效分类器呢？随机平滑通过在输入中注入噪声来获得平滑和更稳健的分类器，从而提供了一种很有前途的框架。在本文中，我们首先证明了随机平滑引入的方差与分类器的另外两个重要性质密切相关，即其Lipschitz常数和边界。更准确地说，我们的工作强调了基分类器的Lipschitz常数对平滑的分类器和经验方差的双重影响。此外，为了增加证明的稳健半径，我们为基分类器引入了不同的单纯形投影技术，以利用Bernstein浓度不等的方差-边际权衡，以及增强的Lipschitz界。实验结果表明，与目前最先进的方法相比，认证的准确率有了显著的提高。我们新的认证程序允许我们使用预先训练的模型，这些模型与随机平滑一起使用，以零射击的方式有效地改进了当前的认证半径。



## **40. SLMIA-SR: Speaker-Level Membership Inference Attacks against Speaker Recognition Systems**

SLMIA-SR：针对说话人识别系统的说话人级别成员推理攻击 cs.CR

In Proceedings of the 31st Network and Distributed System Security  (NDSS) Symposium, 2024

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2309.07983v2) [paper-pdf](http://arxiv.org/pdf/2309.07983v2)

**Authors**: Guangke Chen, Yedi Zhang, Fu Song

**Abstract**: Membership inference attacks allow adversaries to determine whether a particular example was contained in the model's training dataset. While previous works have confirmed the feasibility of such attacks in various applications, none has focused on speaker recognition (SR), a promising voice-based biometric recognition technique. In this work, we propose SLMIA-SR, the first membership inference attack tailored to SR. In contrast to conventional example-level attack, our attack features speaker-level membership inference, i.e., determining if any voices of a given speaker, either the same as or different from the given inference voices, have been involved in the training of a model. It is particularly useful and practical since the training and inference voices are usually distinct, and it is also meaningful considering the open-set nature of SR, namely, the recognition speakers were often not present in the training data. We utilize intra-similarity and inter-dissimilarity, two training objectives of SR, to characterize the differences between training and non-training speakers and quantify them with two groups of features driven by carefully-established feature engineering to mount the attack. To improve the generalizability of our attack, we propose a novel mixing ratio training strategy to train attack models. To enhance the attack performance, we introduce voice chunk splitting to cope with the limited number of inference voices and propose to train attack models dependent on the number of inference voices. Our attack is versatile and can work in both white-box and black-box scenarios. Additionally, we propose two novel techniques to reduce the number of black-box queries while maintaining the attack performance. Extensive experiments demonstrate the effectiveness of SLMIA-SR.

摘要: 成员关系推理攻击允许攻击者确定特定示例是否包含在模型的训练数据集中。虽然以前的工作已经证实了这类攻击在各种应用中的可行性，但还没有人专注于说话人识别(SR)，这是一种很有前途的基于语音的生物识别技术。在这项工作中，我们提出了SLMIA-SR，这是第一个针对SR量身定做的成员推理攻击。与传统的范例级攻击不同，我们的攻击具有说话人级别的成员关系推理，即确定给定说话人的任何声音是否与给定的推理声音相同或不同，参与了模型的训练。它特别有用和实用，因为训练和推理的声音通常是不同的，而且考虑到SR的开放集性质，即识别说话人通常不在训练数据中，这也是有意义的。我们利用随机共振的内相似和互异两个训练目标来刻画训练说话人和非训练说话人之间的差异，并在精心建立的特征工程的驱动下用两组特征来量化它们来发动攻击。为了提高攻击的泛化能力，我们提出了一种新的混合比训练策略来训练攻击模型。为了提高攻击性能，我们引入了语音块分裂来应对有限的推理语音，并提出了根据推理语音的数量来训练攻击模型。我们的攻击是多才多艺的，可以在白盒和黑盒场景中工作。此外，我们还提出了两种新的技术来在保持攻击性能的同时减少黑盒查询的数量。大量实验证明了SLMIA-SR的有效性。



## **41. RetouchUAA: Unconstrained Adversarial Attack via Image Retouching**

RetouchUAA：基于图像修饰的无约束对抗性攻击 cs.CV

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.16478v1) [paper-pdf](http://arxiv.org/pdf/2311.16478v1)

**Authors**: Mengda Xie, Yiling He, Meie Fang

**Abstract**: Deep Neural Networks (DNNs) are susceptible to adversarial examples. Conventional attacks generate controlled noise-like perturbations that fail to reflect real-world scenarios and hard to interpretable. In contrast, recent unconstrained attacks mimic natural image transformations occurring in the real world for perceptible but inconspicuous attacks, yet compromise realism due to neglect of image post-processing and uncontrolled attack direction. In this paper, we propose RetouchUAA, an unconstrained attack that exploits a real-life perturbation: image retouching styles, highlighting its potential threat to DNNs. Compared to existing attacks, RetouchUAA offers several notable advantages. Firstly, RetouchUAA excels in generating interpretable and realistic perturbations through two key designs: the image retouching attack framework and the retouching style guidance module. The former custom-designed human-interpretability retouching framework for adversarial attack by linearizing images while modelling the local processing and retouching decision-making in human retouching behaviour, provides an explicit and reasonable pipeline for understanding the robustness of DNNs against retouching. The latter guides the adversarial image towards standard retouching styles, thereby ensuring its realism. Secondly, attributed to the design of the retouching decision regularization and the persistent attack strategy, RetouchUAA also exhibits outstanding attack capability and defense robustness, posing a heavy threat to DNNs. Experiments on ImageNet and Place365 reveal that RetouchUAA achieves nearly 100\% white-box attack success against three DNNs, while achieving a better trade-off between image naturalness, transferability and defense robustness than baseline attacks.

摘要: 深度神经网络(DNN)很容易受到敌意例子的影响。传统的攻击会产生受控的噪声类扰动，无法反映真实世界的场景，很难解释。相比之下，最近的无约束攻击模仿了真实世界中发生的自然图像变换，用于可感知但不明显的攻击，但由于忽略了图像后处理和不受控制的攻击方向，从而损害了真实感。在本文中，我们提出了一种无约束攻击RetouchUAA，它利用了现实生活中的一种扰动：图像修饰风格，强调了它对DNNS的潜在威胁。与现有的攻击相比，RetouchUAA具有几个显著的优势。首先，RetouchUAA通过两个关键设计：图像修饰攻击框架和修饰风格制导模块，擅长生成可解释和逼真的扰动。以往定制的人工可解释修饰框架通过线性化图像，同时模拟人类修饰行为中的局部处理和修饰决策，为理解DNN对修饰的健壮性提供了一条明确而合理的管道。后者引导对抗性的形象走向标准的润色风格，从而确保其现实主义。其次，由于修正决策正则化和持续攻击策略的设计，RetouchUAA还表现出了出色的攻击能力和防御健壮性，对DNN构成了严重威胁。在ImageNet和Place365上的实验表明，RetouchUAA对3个DNN的白盒攻击成功率接近100%，同时在图像自然性、可转移性和防御健壮性之间取得了比基线攻击更好的折衷。



## **42. Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information**

基于困惑度量和上下文信息的令牌级敌意提示检测 cs.CL

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.11509v2) [paper-pdf](http://arxiv.org/pdf/2311.11509v2)

**Authors**: Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Viswanathan Swaminathan

**Abstract**: In recent years, Large Language Models (LLM) have emerged as pivotal tools in various applications. However, these models are susceptible to adversarial prompt attacks, where attackers can carefully curate input strings that lead to undesirable outputs. The inherent vulnerability of LLMs stems from their input-output mechanisms, especially when presented with intensely out-of-distribution (OOD) inputs. This paper proposes a token-level detection method to identify adversarial prompts, leveraging the LLM's capability to predict the next token's probability. We measure the degree of the model's perplexity and incorporate neighboring token information to encourage the detection of contiguous adversarial prompt sequences. As a result, we propose two methods: one that identifies each token as either being part of an adversarial prompt or not, and another that estimates the probability of each token being part of an adversarial prompt.

摘要: 近年来，大型语言模型(LLM)已经成为各种应用中的关键工具。然而，这些模型容易受到敌意提示攻击，攻击者可以仔细策划导致不良输出的输入字符串。低成本管理的内在脆弱性源于其投入-产出机制，特别是在投入严重失配(OOD)的情况下。提出了一种令牌级检测方法来识别敌意提示，利用LLM的能力来预测下一个令牌的概率。我们测量模型的困惑程度，并结合相邻令牌信息来鼓励对连续对抗性提示序列的检测。因此，我们提出了两种方法：一种是识别每个令牌是不是对抗性提示的一部分，另一种是估计每个令牌是对抗性提示的一部分的概率。



## **43. Instruct2Attack: Language-Guided Semantic Adversarial Attacks**

Instruct2Attack：语言制导的语义对抗性攻击 cs.CV

under submission, code coming soon

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.15551v1) [paper-pdf](http://arxiv.org/pdf/2311.15551v1)

**Authors**: Jiang Liu, Chen Wei, Yuxiang Guo, Heng Yu, Alan Yuille, Soheil Feizi, Chun Pong Lau, Rama Chellappa

**Abstract**: We propose Instruct2Attack (I2A), a language-guided semantic attack that generates semantically meaningful perturbations according to free-form language instructions. We make use of state-of-the-art latent diffusion models, where we adversarially guide the reverse diffusion process to search for an adversarial latent code conditioned on the input image and text instruction. Compared to existing noise-based and semantic attacks, I2A generates more natural and diverse adversarial examples while providing better controllability and interpretability. We further automate the attack process with GPT-4 to generate diverse image-specific text instructions. We show that I2A can successfully break state-of-the-art deep neural networks even under strong adversarial defenses, and demonstrate great transferability among a variety of network architectures.

摘要: 我们提出了Instruct2Attack(I2a)，这是一种语言制导的语义攻击，它根据自由形式的语言指令生成语义上有意义的扰动。我们利用最新的潜在扩散模型，对反向扩散过程进行对抗性引导，以搜索以输入图像和文本指令为条件的对抗性潜代码。与现有的基于噪声和语义的攻击相比，i2a生成了更自然和多样化的对抗性示例，同时提供了更好的可控性和可解释性。我们使用GPT-4进一步自动化攻击过程，以生成各种特定于图像的文本指令。我们证明了i2a即使在强大的对手防御下也能成功地破解最先进的深度神经网络，并在各种网络体系结构之间表现出很强的可移植性。



## **44. Confidence Is All You Need for MI Attacks**

信心是MI攻击所需要的全部 cs.LG

2 pages, 1 figure

**SubmitDate**: 2023-11-26    [abs](http://arxiv.org/abs/2311.15373v1) [paper-pdf](http://arxiv.org/pdf/2311.15373v1)

**Authors**: Abhishek Sinha, Himanshi Tibrewal, Mansi Gupta, Nikhar Waghela, Shivank Garg

**Abstract**: In this evolving era of machine learning security, membership inference attacks have emerged as a potent threat to the confidentiality of sensitive data. In this attack, adversaries aim to determine whether a particular point was used during the training of a target model. This paper proposes a new method to gauge a data point's membership in a model's training set. Instead of correlating loss with membership, as is traditionally done, we have leveraged the fact that training examples generally exhibit higher confidence values when classified into their actual class. During training, the model is essentially being 'fit' to the training data and might face particular difficulties in generalization to unseen data. This asymmetry leads to the model achieving higher confidence on the training data as it exploits the specific patterns and noise present in the training data. Our proposed approach leverages the confidence values generated by the machine learning model. These confidence values provide a probabilistic measure of the model's certainty in its predictions and can further be used to infer the membership of a given data point. Additionally, we also introduce another variant of our method that allows us to carry out this attack without knowing the ground truth(true class) of a given data point, thus offering an edge over existing label-dependent attack methods.

摘要: 在这个不断发展的机器学习安全时代，成员身份推理攻击已经成为对敏感数据保密性的有力威胁。在这种攻击中，对手的目标是确定在目标模型的训练过程中是否使用了特定的点。本文提出了一种新的方法来衡量数据点在模型训练集中的隶属度。我们没有像传统上那样将损失与成员关系联系起来，而是利用了这样一个事实，即当分类到实际班级时，训练样本通常显示出更高的置信度。在训练过程中，该模型基本上与训练数据“匹配”，在推广到看不见的数据时可能会面临特别的困难。这种不对称性导致模型在训练数据上实现了更高的置信度，因为它利用了训练数据中存在的特定模式和噪声。我们提出的方法利用了机器学习模型生成的置信度。这些置信值提供了模型在其预测中的确定性的概率度量，并可进一步用于推断给定数据点的成员资格。此外，我们还介绍了我们的方法的另一个变体，它允许我们在不知道给定数据点的基本事实(真类)的情况下执行这种攻击，从而提供了比现有的依赖标签的攻击方法更好的优势。



## **45. Adversarial Purification of Information Masking**

信息掩饰的对抗性净化 cs.CV

**SubmitDate**: 2023-11-26    [abs](http://arxiv.org/abs/2311.15339v1) [paper-pdf](http://arxiv.org/pdf/2311.15339v1)

**Authors**: Sitong Liu, Zhichao Lian, Shuangquan Zhang, Liang Xiao

**Abstract**: Adversarial attacks meticulously generate minuscule, imperceptible perturbations to images to deceive neural networks. Counteracting these, adversarial purification methods seek to transform adversarial input samples into clean output images to defend against adversarial attacks. Nonetheless, extent generative models fail to effectively eliminate adversarial perturbations, yielding less-than-ideal purification results. We emphasize the potential threat of residual adversarial perturbations to target models, quantitatively establishing a relationship between perturbation scale and attack capability. Notably, the residual perturbations on the purified image primarily stem from the same-position patch and similar patches of the adversarial sample. We propose a novel adversarial purification approach named Information Mask Purification (IMPure), aims to extensively eliminate adversarial perturbations. To obtain an adversarial sample, we first mask part of the patches information, then reconstruct the patches to resist adversarial perturbations from the patches. We reconstruct all patches in parallel to obtain a cohesive image. Then, in order to protect the purified samples against potential similar regional perturbations, we simulate this risk by randomly mixing the purified samples with the input samples before inputting them into the feature extraction network. Finally, we establish a combined constraint of pixel loss and perceptual loss to augment the model's reconstruction adaptability. Extensive experiments on the ImageNet dataset with three classifier models demonstrate that our approach achieves state-of-the-art results against nine adversarial attack methods. Implementation code and pre-trained weights can be accessed at \textcolor{blue}{https://github.com/NoWindButRain/IMPure}.

摘要: 敌意攻击小心翼翼地对图像产生微小的、不可察觉的扰动，以欺骗神经网络。对抗性净化方法寻求将对抗性输入样本转换为干净的输出图像以防御对抗性攻击。然而，广度生成模型不能有效地消除对抗性扰动，产生不太理想的纯化结果。我们强调了残留对抗性扰动对目标模型的潜在威胁，定量地建立了扰动规模与攻击能力之间的关系。值得注意的是，纯化图像上的残留扰动主要源于对抗性样本的相同位置补丁和相似补丁。我们提出了一种新的对抗性净化方法，称为信息掩码净化(INPURE)，旨在广泛地消除对抗性扰动。为了获得对抗性样本，我们首先掩蔽部分斑块信息，然后重建斑块以抵抗来自斑块的对抗性扰动。我们对所有的块进行并行重建，以获得一个连贯的图像。然后，为了保护纯化样本不受潜在的相似区域扰动的影响，在输入到特征提取网络之前，我们通过将纯化样本与输入样本随机混合来模拟这种风险。最后，我们建立了像素损失和感知损失的组合约束，以增强模型的重建适应性。在具有三种分类器模型的ImageNet数据集上的大量实验表明，该方法对九种对抗性攻击方法取得了最先进的结果。实施代码和预先训练的权重可在\textcolor{blue}{https://github.com/NoWindButRain/IMPure}.上访问



## **46. Robust Graph Neural Networks via Unbiased Aggregation**

基于无偏聚集的鲁棒图神经网络 cs.LG

**SubmitDate**: 2023-11-25    [abs](http://arxiv.org/abs/2311.14934v1) [paper-pdf](http://arxiv.org/pdf/2311.14934v1)

**Authors**: Ruiqi Feng, Zhichao Hou, Tyler Derr, Xiaorui Liu

**Abstract**: The adversarial robustness of Graph Neural Networks (GNNs) has been questioned due to the false sense of security uncovered by strong adaptive attacks despite the existence of numerous defenses. In this work, we delve into the robustness analysis of representative robust GNNs and provide a unified robust estimation point of view to understand their robustness and limitations. Our novel analysis of estimation bias motivates the design of a robust and unbiased graph signal estimator. We then develop an efficient Quasi-Newton iterative reweighted least squares algorithm to solve the estimation problem, which unfolds as robust unbiased aggregation layers in GNNs with a theoretical convergence guarantee. Our comprehensive experiments confirm the strong robustness of our proposed model, and the ablation study provides a deep understanding of its advantages.

摘要: 尽管存在大量的防御措施，但由于强自适应攻击所揭示的虚假安全感，图神经网络(GNN)的对抗健壮性受到了质疑。在这项工作中，我们深入研究了具有代表性的稳健GNN的稳健性分析，并提供了一个统一的稳健估计的观点来理解它们的稳健性和局限性。我们对估计偏差的新颖分析激发了稳健和无偏图信号估计器的设计。然后，我们提出了一种有效的拟牛顿迭代重加权最小二乘算法来解决估计问题，该算法在理论上保证收敛的情况下表现为GNN中健壮的无偏聚合层。我们的综合实验证实了我们所提出的模型具有很强的稳健性，消融研究使我们对其优势有了更深的理解。



## **47. Exploiting Large Language Models (LLMs) through Deception Techniques and Persuasion Principles**

通过欺骗技术和说服原则开发大型语言模型（LLM） cs.HC

10 pages, 16 tables, 5 figures, IEEE BigData 2023 (Workshops)

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14876v1) [paper-pdf](http://arxiv.org/pdf/2311.14876v1)

**Authors**: Sonali Singh, Faranak Abri, Akbar Siami Namin

**Abstract**: With the recent advent of Large Language Models (LLMs), such as ChatGPT from OpenAI, BARD from Google, Llama2 from Meta, and Claude from Anthropic AI, gain widespread use, ensuring their security and robustness is critical. The widespread use of these language models heavily relies on their reliability and proper usage of this fascinating technology. It is crucial to thoroughly test these models to not only ensure its quality but also possible misuses of such models by potential adversaries for illegal activities such as hacking. This paper presents a novel study focusing on exploitation of such large language models against deceptive interactions. More specifically, the paper leverages widespread and borrows well-known techniques in deception theory to investigate whether these models are susceptible to deceitful interactions.   This research aims not only to highlight these risks but also to pave the way for robust countermeasures that enhance the security and integrity of language models in the face of sophisticated social engineering tactics. Through systematic experiments and analysis, we assess their performance in these critical security domains. Our results demonstrate a significant finding in that these large language models are susceptible to deception and social engineering attacks.

摘要: 随着大型语言模型(LLM)的出现，如OpenAI的ChatGPT、Google的Bard、Meta的Llama2和Anthropic AI的Claude，获得了广泛的使用，确保它们的安全性和健壮性至关重要。这些语言模型的广泛使用在很大程度上依赖于它们的可靠性和对这项迷人技术的正确使用。至关重要的是，彻底测试这些模型，不仅要确保其质量，还要确保潜在对手可能将这些模型滥用于黑客等非法活动。本文提出了一项新的研究，重点是利用如此大的语言模型来对抗欺骗性交互。更具体地说，本文利用广泛使用的欺骗理论中的著名技术来调查这些模型是否容易受到欺骗性交互作用的影响。这项研究不仅旨在强调这些风险，而且还为在复杂的社会工程策略面前增强语言模型的安全性和完整性的稳健对策铺平道路。通过系统的实验和分析，我们评估了它们在这些关键安全域中的性能。我们的结果证明了一个重要的发现，即这些大型语言模型容易受到欺骗和社会工程攻击。



## **48. Adversarial Machine Learning in Latent Representations of Neural Networks**

神经网络潜在表示中的对抗性机器学习 cs.LG

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2309.17401v2) [paper-pdf](http://arxiv.org/pdf/2309.17401v2)

**Authors**: Milin Zhang, Mohammad Abdi, Francesco Restuccia

**Abstract**: Distributed deep neural networks (DNNs) have been shown to reduce the computational burden of mobile devices and decrease the end-to-end inference latency in edge computing scenarios. While distributed DNNs have been studied, to the best of our knowledge the resilience of distributed DNNs to adversarial action still remains an open problem. In this paper, we fill the existing research gap by rigorously analyzing the robustness of distributed DNNs against adversarial action. We cast this problem in the context of information theory and introduce two new measurements for distortion and robustness. Our theoretical findings indicate that (i) assuming the same level of information distortion, latent features are always more robust than input representations; (ii) the adversarial robustness is jointly determined by the feature dimension and the generalization capability of the DNN. To test our theoretical findings, we perform extensive experimental analysis by considering 6 different DNN architectures, 6 different approaches for distributed DNN and 10 different adversarial attacks to the ImageNet-1K dataset. Our experimental results support our theoretical findings by showing that the compressed latent representations can reduce the success rate of adversarial attacks by 88% in the best case and by 57% on the average compared to attacks to the input space.

摘要: 分布式深度神经网络可以减轻移动设备的计算负担，减少边缘计算场景中的端到端推理延迟。虽然已经对分布式DNN进行了研究，但就我们所知，分布式DNN对敌意行为的恢复能力仍然是一个悬而未决的问题。在本文中，我们通过严格分析分布式DNN对攻击行为的健壮性来填补现有的研究空白。我们把这个问题放在信息论的背景下，并引入了两个新的失真和稳健性度量。我们的理论结果表明：(I)假设信息失真程度相同，潜在特征总是比输入表示更健壮；(Ii)DNN的对抗健壮性由特征维度和泛化能力共同决定。为了验证我们的理论发现，我们通过考虑6种不同的DNN体系结构、6种不同的分布式DNN方法和10种不同的针对ImageNet-1K数据集的对手攻击进行了广泛的实验分析。我们的实验结果支持我们的理论发现，与对输入空间的攻击相比，压缩的潜在表示在最好的情况下可以使对抗性攻击的成功率降低88%，平均降低57%。



## **49. Tamper-Evident Pairing**

防篡改配对 cs.CR

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14790v1) [paper-pdf](http://arxiv.org/pdf/2311.14790v1)

**Authors**: Aleksandar Manev

**Abstract**: Establishing a secure connection between wireless devices has become significantly important with the increasing number of Wi-Fi products coming to the market. In order to provide an easy and secure pairing standard, the Wi-Fi Alliance has designed the Wi-Fi Protected Setup. Push-Button Configuration (PBC) is part of this standard and is especially useful for pairing devices with physical limitations. However, PBC is proven to be vulnerable to man-in-the-middle (MITM) attacks. Tamper-Evident Pairing (TEP) is an improvement of the PBC standard, which aims to fix the MITM vulnerability without interfering the useful properties of PBC. It relies on the Tamper-Evident Announcement (TEA), which guarantees that an adversary can neither tamper a transmitted message without being detected, nor hide the fact that the message has been sent. The security properties of TEP were proven manually by its authors and tested with the Uppaal and Spin model checkers. During the Uppaal model checking, no vulnerabilities were found. However, the Spin model revealed a case, in which the TEP's security is not guaranteed. In this paper, we first provide a comprehensive overview of the TEP protocol, including all information needed to understand how it works. Furthermore, we summarize the security checks performed on it, give the circumstances, under which it is no longer resistant to MITM attacks and explain the reasons why they could not be revealed with the first model. Nevertheless, future work is required to gain full certainty of the TEP's security before applying it in the industry.

摘要: 随着越来越多的Wi-Fi产品进入市场，在无线设备之间建立安全连接变得非常重要。为了提供简单安全的配对标准，Wi-Fi联盟设计了Wi-Fi保护设置。按钮配置(PBC)是该标准的一部分，对于具有物理限制的配对设备特别有用。然而，事实证明，PBC容易受到中间人(MITM)攻击。篡改明显配对(TEP)是对PBC标准的改进，旨在修复MITM漏洞而不干扰PBC的有用特性。它依赖于明显篡改声明(TEA)，该声明保证攻击者既不能在不被检测到的情况下篡改传输的消息，也不能隐藏消息已经发送的事实。TEP的安全属性由其作者手动验证，并使用Uppaal和Spin模型检查器进行测试。在Uppaal模型检查过程中，没有发现任何漏洞。然而，SPIN模型揭示了一种情况，在这种情况下，TEP的安全性得不到保证。在本文中，我们首先全面概述TEP协议，包括了解其工作原理所需的所有信息。此外，我们总结了对其进行的安全检查，给出了它不再抵抗MITM攻击的情况，并解释了第一种模型无法揭示它们的原因。然而，在将TEP应用于行业之前，需要进行进一步的工作，以完全确定TEP的安全性。



## **50. Mind the box: $l_1$-APGD for sparse adversarial attacks on image classifiers**

注意：$L_1$-针对图像分类器的稀疏对抗性攻击的APGD cs.LG

In ICML 2021. Fixed typos in Eq. (3) and Eq. (4)

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2103.01208v3) [paper-pdf](http://arxiv.org/pdf/2103.01208v3)

**Authors**: Francesco Croce, Matthias Hein

**Abstract**: We show that when taking into account also the image domain $[0,1]^d$, established $l_1$-projected gradient descent (PGD) attacks are suboptimal as they do not consider that the effective threat model is the intersection of the $l_1$-ball and $[0,1]^d$. We study the expected sparsity of the steepest descent step for this effective threat model and show that the exact projection onto this set is computationally feasible and yields better performance. Moreover, we propose an adaptive form of PGD which is highly effective even with a small budget of iterations. Our resulting $l_1$-APGD is a strong white-box attack showing that prior works overestimated their $l_1$-robustness. Using $l_1$-APGD for adversarial training we get a robust classifier with SOTA $l_1$-robustness. Finally, we combine $l_1$-APGD and an adaptation of the Square Attack to $l_1$ into $l_1$-AutoAttack, an ensemble of attacks which reliably assesses adversarial robustness for the threat model of $l_1$-ball intersected with $[0,1]^d$.

摘要: 我们证明了当同时考虑象域$[0，1]^d$时，已建立的$L_1$投影梯度下降攻击是次优的，因为它们没有考虑有效的威胁模型是$L_1$球和$[0，1]^d$的交集。我们研究了这一有效威胁模型的最陡下降步长的期望稀疏性，并证明了在该集合上的精确投影在计算上是可行的，并且产生了更好的性能。此外，我们提出了一种自适应形式的PGD，它即使在迭代预算很小的情况下也是非常有效的。我们得到的$L_1$-APGD是一个强白盒攻击，表明以前的工作高估了它们的$L_1$-稳健性。利用$L_1$-APGD进行对抗性训练，得到一个具有SOTA$L_1$-健壮性的稳健分类器。最后，我们将$L_1$-APGD和对$L_1$的方形攻击的改编合并为$L_1$-AutoAttack，这是一个攻击集合，它可靠地评估了$L_1$球与$[0，1]^d$相交的威胁模型的对手健壮性。



