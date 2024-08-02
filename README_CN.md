# Latest Adversarial Attack Papers
**update at 2024-08-02 16:08:23**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Tamper-Resistant Safeguards for Open-Weight LLMs**

开放重量LLM的防篡改保障措施 cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00761v1) [paper-pdf](http://arxiv.org/pdf/2408.00761v1)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after thousands of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that tamper-resistance is a tractable problem, opening up a promising new avenue to improve the safety and security of open-weight LLMs.

摘要: 大型语言模型(LLM)功能的快速发展引起了人们对其潜在恶意使用的广泛关注。开放重量LLM提出了独特的挑战，因为现有的保障措施缺乏对篡改模型权重的篡改攻击的稳健性。例如，最近的研究表明，通过几个步骤的微调，就可以很容易地消除拒绝和遗忘的保障措施。这些漏洞需要新的方法来实现安全释放未加重量的低密度脂蛋白。我们开发了一种名为TAR的方法，用于在开放重量的LLM中构建防篡改保护措施，以便对手即使在数千个步骤的微调之后也无法移除这些保护措施。在广泛的评估和红团队分析中，我们发现我们的方法在保持良性性能的同时大大提高了防篡改能力。我们的结果表明，防篡改是一个容易解决的问题，为提高开重LLMS的安全性开辟了一条很有前途的新途径。



## **2. CERT-ED: Certifiably Robust Text Classification for Edit Distance**

CERT-ED：针对编辑距离的可认证鲁棒文本分类 cs.CL

22 pages, 3 figures, 12 tables. Include 11 pages of appendices

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00728v1) [paper-pdf](http://arxiv.org/pdf/2408.00728v1)

**Authors**: Zhuoqun Huang, Neil G Marchant, Olga Ohrimenko, Benjamin I. P. Rubinstein

**Abstract**: With the growing integration of AI in daily life, ensuring the robustness of systems to inference-time attacks is crucial. Among the approaches for certifying robustness to such adversarial examples, randomized smoothing has emerged as highly promising due to its nature as a wrapper around arbitrary black-box models. Previous work on randomized smoothing in natural language processing has primarily focused on specific subsets of edit distance operations, such as synonym substitution or word insertion, without exploring the certification of all edit operations. In this paper, we adapt Randomized Deletion (Huang et al., 2023) and propose, CERTified Edit Distance defense (CERT-ED) for natural language classification. Through comprehensive experiments, we demonstrate that CERT-ED outperforms the existing Hamming distance method RanMASK (Zeng et al., 2023) in 4 out of 5 datasets in terms of both accuracy and the cardinality of the certificate. By covering various threat models, including 5 direct and 5 transfer attacks, our method improves empirical robustness in 38 out of 50 settings.

摘要: 随着人工智能在日常生活中的日益融合，确保系统对推理时攻击的健壮性至关重要。在证明对这种对抗性例子的稳健性的方法中，随机平滑因其作为任意黑盒模型的包装器的性质而变得非常有前途。以前关于自然语言处理中的随机化平滑的工作主要集中在编辑距离操作的特定子集上，例如同义词替换或单词插入，而没有探索所有编辑操作的认证。在本文中，我们采用了随机化删除(Huang等人，2023)，并提出了用于自然语言分类的认证编辑距离防御(CERT-ED)。通过综合实验，我们证明了CERT-ED在5个数据集中的4个上优于现有的Hamming距离方法RanMASK(Zeng等人，2023)，无论是在准确率上还是在证书的基数上。通过覆盖各种威胁模型，包括5个直接攻击和5个转移攻击，我们的方法在50种设置中的38种情况下提高了经验稳健性。



## **3. Prover-Verifier Games improve legibility of LLM outputs**

证明者-验证者游戏提高了LLM输出的清晰度 cs.CL

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2407.13692v2) [paper-pdf](http://arxiv.org/pdf/2407.13692v2)

**Authors**: Jan Hendrik Kirchner, Yining Chen, Harri Edwards, Jan Leike, Nat McAleese, Yuri Burda

**Abstract**: One way to increase confidence in the outputs of Large Language Models (LLMs) is to support them with reasoning that is clear and easy to check -- a property we call legibility. We study legibility in the context of solving grade-school math problems and show that optimizing chain-of-thought solutions only for answer correctness can make them less legible. To mitigate the loss in legibility, we propose a training algorithm inspired by Prover-Verifier Game from Anil et al. (2021). Our algorithm iteratively trains small verifiers to predict solution correctness, "helpful" provers to produce correct solutions that the verifier accepts, and "sneaky" provers to produce incorrect solutions that fool the verifier. We find that the helpful prover's accuracy and the verifier's robustness to adversarial attacks increase over the course of training. Furthermore, we show that legibility training transfers to time-constrained humans tasked with verifying solution correctness. Over course of LLM training human accuracy increases when checking the helpful prover's solutions, and decreases when checking the sneaky prover's solutions. Hence, training for checkability by small verifiers is a plausible technique for increasing output legibility. Our results suggest legibility training against small verifiers as a practical avenue for increasing legibility of large LLMs to humans, and thus could help with alignment of superhuman models.

摘要: 增加对大型语言模型(LLM)输出的信心的一种方法是用清晰且易于检查的推理来支持它们--我们称之为易读性。我们在解决小学数学问题的背景下研究了易读性，并表明只为了答案的正确性而优化思维链解决方案会降低它们的易读性。为了减少易读性的损失，我们提出了一种受Anil等人的Prover-Verator游戏启发的训练算法。(2021年)。我们的算法迭代地训练小的验证者来预测解决方案的正确性，“有帮助的”验证者来产生验证者接受的正确的解决方案，而“偷偷摸摸”的验证者产生愚弄验证者的不正确的解决方案。我们发现，随着训练过程的进行，有益证明者的准确率和验证者对敌意攻击的健壮性都有所提高。此外，我们还表明，易读性训练转移到负责验证解决方案正确性的时间受限的人身上。在LLM训练过程中，当检查有用的证明者的解时，人类的准确率提高，而当检查偷偷摸摸的证明者的解时，人类的准确率降低。因此，由小型验证员进行可校验性培训是提高输出清晰度的一种可行的技术。我们的结果表明，针对小验证者的易读性训练是提高大型LLM对人类易读性的实用途径，因此可能有助于超人模型的对齐。



## **4. Defending Jailbreak Attack in VLMs via Cross-modality Information Detector**

通过跨模式信息检测器防御VLM中的越狱攻击 cs.CL

12 pages, 9 figures

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2407.21659v2) [paper-pdf](http://arxiv.org/pdf/2407.21659v2)

**Authors**: Yue Xu, Xiuyuan Qi, Zhan Qin, Wenjie Wang

**Abstract**: Vision Language Models (VLMs) extend the capacity of LLMs to comprehensively understand vision information, achieving remarkable performance in many vision-centric tasks. Despite that, recent studies have shown that these models are susceptible to jailbreak attacks, which refer to an exploitative technique where malicious users can break the safety alignment of the target model and generate misleading and harmful answers. This potential threat is caused by both the inherent vulnerabilities of LLM and the larger attack scope introduced by vision input. To enhance the security of VLMs against jailbreak attacks, researchers have developed various defense techniques. However, these methods either require modifications to the model's internal structure or demand significant computational resources during the inference phase. Multimodal information is a double-edged sword. While it increases the risk of attacks, it also provides additional data that can enhance safeguards. Inspired by this, we propose $\underline{\textbf{C}}$ross-modality $\underline{\textbf{I}}$nformation $\underline{\textbf{DE}}$tecto$\underline{\textbf{R}}$ ($\textit{CIDER})$, a plug-and-play jailbreaking detector designed to identify maliciously perturbed image inputs, utilizing the cross-modal similarity between harmful queries and adversarial images. This simple yet effective cross-modality information detector, $\textit{CIDER}$, is independent of the target VLMs and requires less computation cost. Extensive experimental results demonstrate the effectiveness and efficiency of $\textit{CIDER}$, as well as its transferability to both white-box and black-box VLMs.

摘要: 视觉语言模型扩展了视觉语言模型全面理解视觉信息的能力，在许多以视觉为中心的任务中取得了显著的表现。尽管如此，最近的研究表明，这些模型容易受到越狱攻击，越狱攻击指的是一种利用技术，恶意用户可以破坏目标模型的安全对齐，并生成误导性和有害的答案。这种潜在的威胁既是由LLM固有的漏洞造成的，也是由视觉输入引入的更大的攻击范围造成的。为了增强VLMS抵御越狱攻击的安全性，研究人员开发了各种防御技术。然而，这些方法要么需要修改模型的内部结构，要么在推理阶段需要大量的计算资源。多式联运信息是一把双刃剑。虽然它增加了攻击的风险，但它也提供了额外的数据，可以加强安全措施。受此启发，我们提出了一种即插即用的越狱检测器，旨在利用有害查询和敌意图像之间的跨模式相似性来识别恶意干扰的图像输入。这种简单而有效的跨通道信息检测器，$\textit{cider}$，独立于目标VLM，并且需要更少的计算成本。大量的实验结果证明了该算法的有效性和高效性，以及它对白盒和黑盒VLM的可转换性。



## **5. Autonomous LLM-Enhanced Adversarial Attack for Text-to-Motion**

针对文本到运动的自主LLM增强对抗攻击 cs.CV

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00352v1) [paper-pdf](http://arxiv.org/pdf/2408.00352v1)

**Authors**: Honglei Miao, Fan Ma, Ruijie Quan, Kun Zhan, Yi Yang

**Abstract**: Human motion generation driven by deep generative models has enabled compelling applications, but the ability of text-to-motion (T2M) models to produce realistic motions from text prompts raises security concerns if exploited maliciously. Despite growing interest in T2M, few methods focus on safeguarding these models against adversarial attacks, with existing work on text-to-image models proving insufficient for the unique motion domain. In the paper, we propose ALERT-Motion, an autonomous framework leveraging large language models (LLMs) to craft targeted adversarial attacks against black-box T2M models. Unlike prior methods modifying prompts through predefined rules, ALERT-Motion uses LLMs' knowledge of human motion to autonomously generate subtle yet powerful adversarial text descriptions. It comprises two key modules: an adaptive dispatching module that constructs an LLM-based agent to iteratively refine and search for adversarial prompts; and a multimodal information contrastive module that extracts semantically relevant motion information to guide the agent's search. Through this LLM-driven approach, ALERT-Motion crafts adversarial prompts querying victim models to produce outputs closely matching targeted motions, while avoiding obvious perturbations. Evaluations across popular T2M models demonstrate ALERT-Motion's superiority over previous methods, achieving higher attack success rates with stealthier adversarial prompts. This pioneering work on T2M adversarial attacks highlights the urgency of developing defensive measures as motion generation technology advances, urging further research into safe and responsible deployment.

摘要: 由深度生成模型驱动的人类运动生成已经实现了令人信服的应用，但文本到运动(T2M)模型从文本提示生成逼真运动的能力如果被恶意利用，会引发安全问题。尽管对T2M的兴趣与日俱增，但很少有方法专注于保护这些模型免受对手攻击，现有的文本到图像模型的工作被证明不足以满足独特的运动域。在本文中，我们提出了ALERT-Motion，这是一个利用大型语言模型(LLM)来针对黑盒T2M模型进行有针对性的对抗性攻击的自主框架。与以前通过预定义规则修改提示的方法不同，ALERT-Motion使用LLMS对人体运动的知识来自主生成微妙但强大的对抗性文本描述。它包括两个关键模块：自适应调度模块和多通道信息对比模块，自适应调度模块构建了一个基于LLM的代理，用于迭代地提炼和搜索对手提示；多通道信息对比模块提取语义相关的运动信息来指导代理的搜索。通过这种LLM驱动的方法，ALERT-Motion恶意提示查询受害者模型以产生与目标运动紧密匹配的输出，同时避免明显的扰动。对流行的T2M模型的评估表明，Alert-Motion比以前的方法更具优势，通过更隐蔽的对手提示实现了更高的攻击成功率。这项关于T2M对抗性攻击的开创性工作突显了随着动作生成技术的进步而开发防御措施的紧迫性，促使对安全和负责任的部署进行进一步研究。



## **6. Securing the Diagnosis of Medical Imaging: An In-depth Analysis of AI-Resistant Attacks**

确保医学成像诊断：深入分析抗人工智能攻击 cs.CR

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00348v1) [paper-pdf](http://arxiv.org/pdf/2408.00348v1)

**Authors**: Angona Biswas, MD Abdullah Al Nasim, Kishor Datta Gupta, Roy George, Abdur Rashid

**Abstract**: Machine learning (ML) is a rapidly developing area of medicine that uses significant resources to apply computer science and statistics to medical issues. ML's proponents laud its capacity to handle vast, complicated, and erratic medical data. It's common knowledge that attackers might cause misclassification by deliberately creating inputs for machine learning classifiers. Research on adversarial examples has been extensively conducted in the field of computer vision applications. Healthcare systems are thought to be highly difficult because of the security and life-or-death considerations they include, and performance accuracy is very important. Recent arguments have suggested that adversarial attacks could be made against medical image analysis (MedIA) technologies because of the accompanying technology infrastructure and powerful financial incentives. Since the diagnosis will be the basis for important decisions, it is essential to assess how strong medical DNN tasks are against adversarial attacks. Simple adversarial attacks have been taken into account in several earlier studies. However, DNNs are susceptible to more risky and realistic attacks. The present paper covers recent proposed adversarial attack strategies against DNNs for medical imaging as well as countermeasures. In this study, we review current techniques for adversarial imaging attacks, detections. It also encompasses various facets of these techniques and offers suggestions for the robustness of neural networks to be improved in the future.

摘要: 机器学习(ML)是一个快速发展的医学领域，它利用大量资源将计算机科学和统计学应用于医学问题。ML的支持者称赞其处理海量、复杂和不稳定的医疗数据的能力。众所周知，攻击者可能会通过故意为机器学习分类器创建输入而导致错误分类。对抗性例子的研究在计算机视觉应用领域得到了广泛的开展。医疗保健系统被认为是非常困难的，因为它们包括安全和生死攸关的考虑，而性能的准确性非常重要。最近的论点表明，可以对医学图像分析(媒体)技术进行对抗性攻击，因为伴随而来的是技术基础设施和强大的财政激励。由于诊断将是重要决策的基础，因此评估医疗DNN任务对抗对手攻击的能力有多强是至关重要的。在早期的几项研究中，简单的对抗性攻击已经被考虑在内。然而，DNN更容易受到风险更高、更现实的攻击。本文讨论了最近提出的针对医学成像领域的DNN的对抗性攻击策略以及对策。在这项研究中，我们回顾了现有的对抗性成像攻击和检测技术。它还涵盖了这些技术的各个方面，并为未来神经网络的健壮性提供了建议。



## **7. MAARS: Multi-Rate Attack-Aware Randomized Scheduling for Securing Real-time Systems**

MAARS：用于保护实时系统的多速率攻击感知随机调度 eess.SY

12 pages including references, Total 10 figures (with 3 having  subfigures). This paper was rejected in RTSS 2024 Conference

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00341v1) [paper-pdf](http://arxiv.org/pdf/2408.00341v1)

**Authors**: Arkaprava Sain, Sunandan Adhikary, Ipsita Koley, Soumyajit Dey

**Abstract**: Modern Cyber-Physical Systems (CPSs) consist of numerous control units interconnected by communication networks. Each control unit executes multiple safety-critical and non-critical tasks in real-time. Most of the safety-critical tasks are executed with a fixed sampling period to ensure deterministic timing behaviour that helps in its safety and performance analysis. However, adversaries can exploit this deterministic behaviour of safety-critical tasks to launch inference-based-based attacks on them. This paper aims to prevent and minimize the possibility of such timing inference or schedule-based attacks to compromise the control units. This is done by switching between strategically chosen execution rates of the safety-critical control tasks such that their performance remains unhampered. Thereafter, we present a novel schedule vulnerability analysis methodology to switch between valid schedules generated for these multiple periodicities of the control tasks in run time. Utilizing these strategies, we introduce a novel Multi-Rate Attack-Aware Randomized Scheduling (MAARS) framework for preemptive fixed-priority schedulers that minimize the success rate of timing-inference-based attacks on safety-critical real-time systems. To our knowledge, this is the first work to propose a schedule randomization method with attack awareness that preserves both the control and scheduling aspects. The efficacy of the framework in terms of attack prevention is finally evaluated on several automotive benchmarks in a Hardware-in-loop (HiL) environment.

摘要: 现代网络物理系统(CPSS)由多个控制单元通过通信网络相互连接而成。每个控制单元实时执行多个安全关键和非关键任务。大多数安全关键任务都以固定的采样周期执行，以确保确定性的定时行为，这有助于其安全和性能分析。然而，攻击者可以利用安全关键任务的这种确定性行为来对它们发起基于推理的攻击。本文的目的是防止和最大限度地减少这种定时推断或基于调度的攻击危害控制单元的可能性。这是通过在战略上选择的安全关键控制任务的执行率之间切换来实现的，从而使它们的性能保持不受阻碍。在此基础上，我们提出了一种新的调度脆弱性分析方法，用于在运行时为控制任务的这些多周期生成的有效调度之间进行切换。利用这些策略，我们提出了一种新的多速率攻击感知随机调度(MAARS)框架，用于抢占式固定优先级调度器，以最大限度地降低基于定时推理的攻击对安全关键实时系统的成功率。据我们所知，这是首次提出一种具有攻击感知的调度随机化方法，该方法同时保留了控制和调度两个方面。最后在硬件在环(HIL)环境下的几个汽车基准上对该框架在攻击预防方面的有效性进行了评估。



## **8. OTAD: An Optimal Transport-Induced Robust Model for Agnostic Adversarial Attack**

Ospel：不可知对抗攻击的最佳传输诱导鲁棒模型 cs.LG

14 pages, 2 figures

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00329v1) [paper-pdf](http://arxiv.org/pdf/2408.00329v1)

**Authors**: Kuo Gai, Sicong Wang, Shihua Zhang

**Abstract**: Deep neural networks (DNNs) are vulnerable to small adversarial perturbations of the inputs, posing a significant challenge to their reliability and robustness. Empirical methods such as adversarial training can defend against particular attacks but remain vulnerable to more powerful attacks. Alternatively, Lipschitz networks provide certified robustness to unseen perturbations but lack sufficient expressive power. To harness the advantages of both approaches, we design a novel two-step Optimal Transport induced Adversarial Defense (OTAD) model that can fit the training data accurately while preserving the local Lipschitz continuity. First, we train a DNN with a regularizer derived from optimal transport theory, yielding a discrete optimal transport map linking data to its features. By leveraging the map's inherent regularity, we interpolate the map by solving the convex integration problem (CIP) to guarantee the local Lipschitz property. OTAD is extensible to diverse architectures of ResNet and Transformer, making it suitable for complex data. For efficient computation, the CIP can be solved through training neural networks. OTAD opens a novel avenue for developing reliable and secure deep learning systems through the regularity of optimal transport maps. Empirical results demonstrate that OTAD can outperform other robust models on diverse datasets.

摘要: 深度神经网络(DNN)容易受到输入的微小对抗性扰动，对其可靠性和健壮性提出了重大挑战。经验方法，如对抗性训练，可以防御特定的攻击，但仍然容易受到更强大的攻击。或者，Lipschitz网络提供了对看不见的扰动的公认的稳健性，但缺乏足够的表达能力。为了利用这两种方法的优点，我们设计了一种新的两步最优传输诱导对抗防御(OTAD)模型，该模型在保持局部Lipschitz连续性的同时，能够准确地拟合训练数据。首先，我们用最优传输理论中的正则化方法训练DNN，得到一个离散的最优传输映射，将数据与其特征联系起来。利用映射的内在正则性，通过求解凸积分问题(CIP)对映射进行内插，以保证局部Lipschitz性质。OTAD可以扩展到不同的ResNet和Transformer架构，适合于复杂的数据。为了提高计算效率，可以通过训练神经网络来求解CIP。OTAD通过最优运输地图的规律性，为开发可靠和安全的深度学习系统开辟了一条新的途径。实验结果表明，在不同的数据集上，OTAD模型的性能优于其他稳健模型。



## **9. ADBM: Adversarial diffusion bridge model for reliable adversarial purification**

ADBM：用于可靠对抗净化的对抗扩散桥模型 cs.LG

20 pages

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00315v1) [paper-pdf](http://arxiv.org/pdf/2408.00315v1)

**Authors**: Xiao Li, Wenxuan Sun, Huanran Chen, Qiongxiu Li, Yining Liu, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Recently Diffusion-based Purification (DiffPure) has been recognized as an effective defense method against adversarial examples. However, we find DiffPure which directly employs the original pre-trained diffusion models for adversarial purification, to be suboptimal. This is due to an inherent trade-off between noise purification performance and data recovery quality. Additionally, the reliability of existing evaluations for DiffPure is questionable, as they rely on weak adaptive attacks. In this work, we propose a novel Adversarial Diffusion Bridge Model, termed ADBM. ADBM directly constructs a reverse bridge from the diffused adversarial data back to its original clean examples, enhancing the purification capabilities of the original diffusion models. Through theoretical analysis and experimental validation across various scenarios, ADBM has proven to be a superior and robust defense mechanism, offering significant promise for practical applications.

摘要: 最近，基于扩散的纯化（DiffPure）被认为是针对对抗性例子的有效防御方法。然而，我们发现直接使用原始预训练的扩散模型进行对抗性纯化的迪夫Pure是次优的。这是由于噪音净化性能和数据恢复质量之间固有的权衡。此外，现有的DistPure评估的可靠性值得怀疑，因为它们依赖于弱适应性攻击。在这项工作中，我们提出了一种新型的对抗扩散桥模型，称为ADBM。ADBM直接构建了从扩散的对抗数据到其原始干净示例的反向桥梁，增强了原始扩散模型的净化能力。通过各种场景的理论分析和实验验证，ADBM已被证明是一种卓越且强大的防御机制，为实际应用提供了巨大的前景。



## **10. Adversarial Text Rewriting for Text-aware Recommender Systems**

文本感知推荐系统的对抗性文本重写 cs.IR

Accepted for publication at: 33rd ACM International Conference on  Information and Knowledge Management (CIKM 2024). Code and data at:  https://github.com/sejoonoh/ATR

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00312v1) [paper-pdf](http://arxiv.org/pdf/2408.00312v1)

**Authors**: Sejoon Oh, Gaurav Verma, Srijan Kumar

**Abstract**: Text-aware recommender systems incorporate rich textual features, such as titles and descriptions, to generate item recommendations for users. The use of textual features helps mitigate cold-start problems, and thus, such recommender systems have attracted increased attention. However, we argue that the dependency on item descriptions makes the recommender system vulnerable to manipulation by adversarial sellers on e-commerce platforms. In this paper, we explore the possibility of such manipulation by proposing a new text rewriting framework to attack text-aware recommender systems. We show that the rewriting attack can be exploited by sellers to unfairly uprank their products, even though the adversarially rewritten descriptions are perceived as realistic by human evaluators. Methodologically, we investigate two different variations to carry out text rewriting attacks: (1) two-phase fine-tuning for greater attack performance, and (2) in-context learning for higher text rewriting quality. Experiments spanning 3 different datasets and 4 existing approaches demonstrate that recommender systems exhibit vulnerability against the proposed text rewriting attack. Our work adds to the existing literature around the robustness of recommender systems, while highlighting a new dimension of vulnerability in the age of large-scale automated text generation.

摘要: 文本感知推荐系统结合了丰富的文本特征，如标题和描述，以生成针对用户的项目推荐。文本特征的使用有助于缓解冷启动问题，因此，这样的推荐系统吸引了越来越多的关注。然而，我们认为，对商品描述的依赖使得推荐系统容易受到电子商务平台上敌对卖家的操纵。在本文中，我们通过提出一种新的文本重写框架来攻击文本感知推荐系统，从而探索了这种操纵的可能性。我们表明，重写攻击可以被卖家利用来不公平地升级他们的产品，即使人工评估者认为敌意重写的描述是真实的。在方法论上，我们研究了两种不同的文本重写攻击方法：(1)两阶段微调，以获得更好的攻击性能；(2)上下文学习，以获得更高的文本重写质量。在3个不同的数据集和4个现有方法上的实验表明，推荐系统对所提出的文本重写攻击表现出脆弱性。我们的工作增加了关于推荐系统健壮性的现有文献，同时强调了大规模自动文本生成时代的脆弱性的一个新维度。



## **11. Hidden Poison: Machine Unlearning Enables Camouflaged Poisoning Attacks**

隐藏的毒药：机器遗忘导致伪装中毒攻击 cs.LG

**SubmitDate**: 2024-07-31    [abs](http://arxiv.org/abs/2212.10717v2) [paper-pdf](http://arxiv.org/pdf/2212.10717v2)

**Authors**: Jimmy Z. Di, Jack Douglas, Jayadev Acharya, Gautam Kamath, Ayush Sekhari

**Abstract**: We introduce camouflaged data poisoning attacks, a new attack vector that arises in the context of machine unlearning and other settings when model retraining may be induced. An adversary first adds a few carefully crafted points to the training dataset such that the impact on the model's predictions is minimal. The adversary subsequently triggers a request to remove a subset of the introduced points at which point the attack is unleashed and the model's predictions are negatively affected. In particular, we consider clean-label targeted attacks (in which the goal is to cause the model to misclassify a specific test point) on datasets including CIFAR-10, Imagenette, and Imagewoof. This attack is realized by constructing camouflage datapoints that mask the effect of a poisoned dataset.

摘要: 我们引入了伪装数据中毒攻击，这是一种新的攻击载体，出现在机器取消学习和其他可能引发模型再训练的环境中。对手首先在训练数据集中添加一些精心设计的点，以便对模型预测的影响最小。对手随后触发一个请求，以删除所引入点的子集，此时攻击将被释放，模型的预测将受到负面影响。特别是，我们考虑对包括CIFAR-10、Imagenette和Imagewoof在内的数据集进行干净标签定向攻击（目标是导致模型错误分类特定测试点）。这种攻击是通过构建伪装数据点来掩盖有毒数据集的影响来实现的。



## **12. Vera Verto: Multimodal Hijacking Attack**

Vera Verto：多模式劫持攻击 cs.CR

**SubmitDate**: 2024-07-31    [abs](http://arxiv.org/abs/2408.00129v1) [paper-pdf](http://arxiv.org/pdf/2408.00129v1)

**Authors**: Minxing Zhang, Ahmed Salem, Michael Backes, Yang Zhang

**Abstract**: The increasing cost of training machine learning (ML) models has led to the inclusion of new parties to the training pipeline, such as users who contribute training data and companies that provide computing resources. This involvement of such new parties in the ML training process has introduced new attack surfaces for an adversary to exploit. A recent attack in this domain is the model hijacking attack, whereby an adversary hijacks a victim model to implement their own -- possibly malicious -- hijacking tasks. However, the scope of the model hijacking attack is so far limited to the homogeneous-modality tasks. In this paper, we transform the model hijacking attack into a more general multimodal setting, where the hijacking and original tasks are performed on data of different modalities. Specifically, we focus on the setting where an adversary implements a natural language processing (NLP) hijacking task into an image classification model. To mount the attack, we propose a novel encoder-decoder based framework, namely the Blender, which relies on advanced image and language models. Experimental results show that our modal hijacking attack achieves strong performances in different settings. For instance, our attack achieves 94%, 94%, and 95% attack success rate when using the Sogou news dataset to hijack STL10, CIFAR-10, and MNIST classifiers.

摘要: 培训机器学习(ML)模型的成本不断增加，导致培训渠道中纳入了新的参与方，如提供培训数据的用户和提供计算资源的公司。这些新人在ML训练过程中的参与为对手带来了新的攻击面来利用。该领域最近的一种攻击是模型劫持攻击，借此对手劫持受害者模型来执行他们自己的--可能是恶意的--劫持任务。然而，到目前为止，模型劫持攻击的范围仅限于同质通道任务。在本文中，我们将劫持攻击模型转化为更一般的多通道环境，其中劫持和原始任务是在不同通道的数据上执行的。具体地说，我们关注的是敌手将自然语言处理(NLP)劫持任务实现到图像分类模型中的设置。为了发动攻击，我们提出了一种新的基于编解码器的框架，即Blender，它依赖于先进的图像和语言模型。实验结果表明，我们的模式劫持攻击在不同的设置下都取得了很好的性能。例如，当使用搜狗新闻数据集劫持STL10、CIFAR-10和MNIST分类器时，我们的攻击获得了94%、94%和95%的攻击成功率。



## **13. TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization Methods**

TAROT：使用策略优化方法的面向任务的作者混淆 cs.CL

**SubmitDate**: 2024-07-31    [abs](http://arxiv.org/abs/2407.21630v1) [paper-pdf](http://arxiv.org/pdf/2407.21630v1)

**Authors**: Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi

**Abstract**: Authorship obfuscation aims to disguise the identity of an author within a text by altering the writing style, vocabulary, syntax, and other linguistic features associated with the text author. This alteration needs to balance privacy and utility. While strong obfuscation techniques can effectively hide the author's identity, they often degrade the quality and usefulness of the text for its intended purpose. Conversely, maintaining high utility tends to provide insufficient privacy, making it easier for an adversary to de-anonymize the author. Thus, achieving an optimal trade-off between these two conflicting objectives is crucial. In this paper, we propose TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization, a new unsupervised authorship obfuscation method whose goal is to optimize the privacy-utility trade-off by regenerating the entire text considering its downstream utility. Our approach leverages policy optimization as a fine-tuning paradigm over small language models in order to rewrite texts by preserving author identity and downstream task utility. We show that our approach largely reduce the accuracy of attackers while preserving utility. We make our code and models publicly available.

摘要: 作者身份混淆旨在通过改变与文本作者相关的写作风格、词汇、句法和其他语言特征来掩盖作者在文本中的身份。这一改变需要平衡隐私和效用。虽然强大的混淆技术可以有效地隐藏作者的身份，但它们往往会降低文本的质量和对预期目的的有用性。相反，保持高实用性往往会提供不充分的隐私，使对手更容易解除作者的匿名。因此，在这两个相互冲突的目标之间实现最佳权衡至关重要。在本文中，我们提出了一种新的无监督作者身份混淆方法--TAROT：基于策略优化的面向任务的作者身份混淆方法，其目标是通过重新生成考虑下游效用的整个文本来优化隐私和效用之间的权衡。我们的方法利用策略优化作为小语言模型上的微调范式，以便通过保留作者身份和下游任务效用来重写文本。我们表明，我们的方法在很大程度上降低了攻击者的准确性，同时保持了实用性。我们公开我们的代码和模型。



## **14. On the Perturbed States for Transformed Input-robust Reinforcement Learning**

关于转换输入稳健强化学习的扰动状态 cs.LG

12 pages

**SubmitDate**: 2024-07-31    [abs](http://arxiv.org/abs/2408.00023v1) [paper-pdf](http://arxiv.org/pdf/2408.00023v1)

**Authors**: Tung M. Luu, Haeyong Kang, Tri Ton, Thanh Nguyen, Chang D. Yoo

**Abstract**: Reinforcement Learning (RL) agents demonstrating proficiency in a training environment exhibit vulnerability to adversarial perturbations in input observations during deployment. This underscores the importance of building a robust agent before its real-world deployment. To alleviate the challenging point, prior works focus on developing robust training-based procedures, encompassing efforts to fortify the deep neural network component's robustness or subject the agent to adversarial training against potent attacks. In this work, we propose a novel method referred to as \textit{Transformed Input-robust RL (TIRL)}, which explores another avenue to mitigate the impact of adversaries by employing input transformation-based defenses. Specifically, we introduce two principles for applying transformation-based defenses in learning robust RL agents: \textit{(1) autoencoder-styled denoising} to reconstruct the original state and \textit{(2) bounded transformations (bit-depth reduction and vector quantization (VQ))} to achieve close transformed inputs. The transformations are applied to the state before feeding it into the policy network. Extensive experiments on multiple \mujoco environments demonstrate that input transformation-based defenses, \ie, VQ, defend against several adversaries in the state observations.

摘要: 强化学习(RL)代理在训练环境中表现出熟练程度，在部署期间对输入观测中的对抗性扰动表现出脆弱性。这突显了在现实世界部署之前构建一个强大的代理的重要性。为了缓解这一挑战，以前的工作集中于开发基于训练的稳健过程，包括努力加强深度神经网络组件的稳健性或使代理接受针对强大攻击的对抗性训练。在这项工作中，我们提出了一种新的方法，称为Tirl(Transform Input-Robust RL，Tirl)，它探索了另一种方法，通过使用基于输入变换的防御来减轻对手的影响。具体地说，我们介绍了在学习健壮的RL代理时应用基于变换的防御的两个原则：\textit{(1)自动编码式去噪}来重建原始状态和\textit{(2)有界变换(比特深度减少和矢量量化(VQ))}以获得接近变换的输入。在将国家送入策略网络之前，会将这些转换应用于国家。在多个MUJOCO环境上的广泛实验表明，基于输入变换的防御，即VQ，在状态观察中可以防御几个对手。



## **15. DeepBaR: Fault Backdoor Attack on Deep Neural Network Layers**

DeepBaR：深度神经网络层的故障后门攻击 cs.LG

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.21220v1) [paper-pdf](http://arxiv.org/pdf/2407.21220v1)

**Authors**: C. A. Martínez-Mejía, J. Solano, J. Breier, D. Bucko, X. Hou

**Abstract**: Machine Learning using neural networks has received prominent attention recently because of its success in solving a wide variety of computational tasks, in particular in the field of computer vision. However, several works have drawn attention to potential security risks involved with the training and implementation of such networks. In this work, we introduce DeepBaR, a novel approach that implants backdoors on neural networks by faulting their behavior at training, especially during fine-tuning. Our technique aims to generate adversarial samples by optimizing a custom loss function that mimics the implanted backdoors while adding an almost non-visible trigger in the image. We attack three popular convolutional neural network architectures and show that DeepBaR attacks have a success rate of up to 98.30\%. Furthermore, DeepBaR does not significantly affect the accuracy of the attacked networks after deployment when non-malicious inputs are given. Remarkably, DeepBaR allows attackers to choose an input that looks similar to a given class, from a human perspective, but that will be classified as belonging to an arbitrary target class.

摘要: 基于神经网络的机器学习由于能够成功地解决各种计算任务，特别是在计算机视觉领域，近年来受到了广泛的关注。然而，有几项工作提请注意此类网络的培训和实施所涉及的潜在安全风险。在这项工作中，我们引入了DeepBaR，这是一种新的方法，通过在训练时特别是在微调期间对神经网络的行为进行错误检查来在神经网络上植入后门。我们的技术旨在通过优化自定义损失函数来生成敌意样本，该函数模仿植入的后门，同时在图像中添加几乎不可见的触发器。我们对三种流行的卷积神经网络结构进行了攻击，结果表明DeepBaR攻击的成功率高达98.30%。此外，在给出非恶意输入的情况下，DeepBaR不会显著影响部署后被攻击网络的准确性。值得注意的是，DeepBaR允许攻击者选择从人类角度看与给定类相似的输入，但这将被归类为属于任意目标类。



## **16. AI Safety in Practice: Enhancing Adversarial Robustness in Multimodal Image Captioning**

实践中的人工智能安全：增强多模式图像字幕中的对抗鲁棒性 cs.CV

Accepted into KDD 2024 workshop on Ethical AI

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.21174v1) [paper-pdf](http://arxiv.org/pdf/2407.21174v1)

**Authors**: Maisha Binte Rashid, Pablo Rivas

**Abstract**: Multimodal machine learning models that combine visual and textual data are increasingly being deployed in critical applications, raising significant safety and security concerns due to their vulnerability to adversarial attacks. This paper presents an effective strategy to enhance the robustness of multimodal image captioning models against such attacks. By leveraging the Fast Gradient Sign Method (FGSM) to generate adversarial examples and incorporating adversarial training techniques, we demonstrate improved model robustness on two benchmark datasets: Flickr8k and COCO. Our findings indicate that selectively training only the text decoder of the multimodal architecture shows performance comparable to full adversarial training while offering increased computational efficiency. This targeted approach suggests a balance between robustness and training costs, facilitating the ethical deployment of multimodal AI systems across various domains.

摘要: 结合视觉和文本数据的多模式机器学习模型越来越多地被部署在关键应用程序中，由于它们容易受到对抗性攻击，从而引发了严重的安全和安保问题。本文提出了一种有效的策略来增强多模式图像字幕模型对此类攻击的鲁棒性。通过利用快速梯度符号法（FGSM）生成对抗性示例并结合对抗性训练技术，我们在两个基准数据集：Flickr 8k和COCO上展示了改进的模型稳健性。我们的研究结果表明，选择性地仅训练多模式架构的文本解码器显示出与完全对抗训练相当的性能，同时提供更高的计算效率。这种有针对性的方法表明了稳健性和训练成本之间的平衡，促进了多模式人工智能系统在各个领域的道德部署。



## **17. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

人工智能生成图像检测中的漏洞：对抗性攻击的挑战 cs.CV

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.20836v1) [paper-pdf](http://arxiv.org/pdf/2407.20836v1)

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of these AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. For the task of AIGI detection, we propose a new attack containing two main parts. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous models, e.g. transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as frequency-based post-train Bayesian attack, or FPBA. Through FPBA, we show that adversarial attack is truly a real threat to AIGI detectors, because FPBA can deliver successful black-box attacks across models, generators, defense methods, and even evade cross-generator detection, which is a crucial real-world detection scenario.

摘要: 最近在图像合成方面的进步，特别是随着GaN和扩散模型的出现，放大了公众对虚假信息传播的担忧。为了解决这些问题，人们已经提出了许多人工智能生成的图像(AIGI)检测器，并在识别虚假图像方面取得了良好的性能。然而，对这些AIGI检测器的对抗健壮性仍然缺乏系统的了解。本文研究了白盒和黑盒环境下最新的AIGI检测器抵抗敌意攻击的脆弱性，这是迄今为止很少被研究的。针对AIGI检测任务，我们提出了一种包含两个主要部分的新攻击。首先，受真伪图像在频域存在明显差异的启发，在频域下加入扰动，使图像偏离其原有的频率分布。其次，我们探索了代理模型的完全后验分布，以进一步缩小不同模型之间的差距，例如跨CNN和VITS传输对抗性实例。这是通过引入一种新颖的后训练贝叶斯策略来实现的，该策略将单个代理转变为贝叶斯策略，能够使用一个预先训练的代理来模拟不同的受害者模型，而不需要重新训练。我们将我们的方法命名为基于频率的训练后贝叶斯攻击，或称FPBA。通过FPBA，我们证明了敌意攻击是对AIGI检测器的真正威胁，因为FPBA可以跨模型、生成器、防御方法提供成功的黑盒攻击，甚至可以逃避交叉生成器检测，这是现实世界中的一个关键检测场景。



## **18. Attacking Cooperative Multi-Agent Reinforcement Learning by Adversarial Minority Influence**

利用敌对少数派影响攻击合作多智能体强化学习 cs.LG

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2302.03322v3) [paper-pdf](http://arxiv.org/pdf/2302.03322v3)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Yuwei Zheng, Pu Feng, Xin Yu, Aishan Liu, Yaodong Yang, Bo An, Wenjun Wu, Xianglong Liu

**Abstract**: This study probes the vulnerabilities of cooperative multi-agent reinforcement learning (c-MARL) under adversarial attacks, a critical determinant of c-MARL's worst-case performance prior to real-world implementation. Current observation-based attacks, constrained by white-box assumptions, overlook c-MARL's complex multi-agent interactions and cooperative objectives, resulting in impractical and limited attack capabilities. To address these shortcomes, we propose Adversarial Minority Influence (AMI), a practical and strong for c-MARL. AMI is a practical black-box attack and can be launched without knowing victim parameters. AMI is also strong by considering the complex multi-agent interaction and the cooperative goal of agents, enabling a single adversarial agent to unilaterally misleads majority victims to form targeted worst-case cooperation. This mirrors minority influence phenomena in social psychology. To achieve maximum deviation in victim policies under complex agent-wise interactions, our unilateral attack aims to characterize and maximize the impact of the adversary on the victims. This is achieved by adapting a unilateral agent-wise relation metric derived from mutual information, thereby mitigating the adverse effects of victim influence on the adversary. To lead the victims into a jointly detrimental scenario, our targeted attack deceives victims into a long-term, cooperatively harmful situation by guiding each victim towards a specific target, determined through a trial-and-error process executed by a reinforcement learning agent. Through AMI, we achieve the first successful attack against real-world robot swarms and effectively fool agents in simulated environments into collectively worst-case scenarios, including Starcraft II and Multi-agent Mujoco. The source code and demonstrations can be found at: https://github.com/DIG-Beihang/AMI.

摘要: 该研究探讨了协作多智能体强化学习(c-Marl)在对抗攻击下的脆弱性，这是c-Marl在现实世界实现之前最差情况性能的关键决定因素。目前的基于观测的攻击受白盒假设的约束，忽略了c-Marl复杂的多智能体交互和合作目标，导致攻击能力不切实际和有限。针对这些不足，我们提出了一种实用而强大的c-Marl算法--对抗性少数影响算法。AMI是一种实用的黑盒攻击，可以在不知道受害者参数的情况下启动。通过考虑复杂的多智能体相互作用和智能体的合作目标，使单一对抗智能体能够单方面误导大多数受害者形成有针对性的最坏情况合作，AMI也很强大。这反映了社会心理学中的小众影响现象。为了在复杂的智能体相互作用下实现受害者政策的最大偏差，我们的单边攻击旨在刻画和最大化对手对受害者的影响。这是通过采用来自互信息的单边代理关系度量来实现的，从而减轻了受害者影响对对手的不利影响。为了将受害者引导到共同有害的情景中，我们的有针对性的攻击通过引导每个受害者指向特定的目标，将受害者欺骗到长期的、合作有害的情况中，该特定目标是通过由强化学习代理执行的反复试验过程确定的。通过AMI，我们实现了对真实世界机器人群的第一次成功攻击，并有效地将模拟环境中的代理愚弄到了集体最坏的情况下，包括星际争霸II和多代理Mujoco。源代码和演示可在以下网址找到：https://github.com/DIG-Beihang/AMI.



## **19. Prompt-Driven Contrastive Learning for Transferable Adversarial Attacks**

可转移对抗攻击的预算驱动对比学习 cs.CV

Accepted to ECCV 2024, Project Page: https://PDCL-Attack.github.io

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.20657v1) [paper-pdf](http://arxiv.org/pdf/2407.20657v1)

**Authors**: Hunmin Yang, Jongoh Jeong, Kuk-Jin Yoon

**Abstract**: Recent vision-language foundation models, such as CLIP, have demonstrated superior capabilities in learning representations that can be transferable across diverse range of downstream tasks and domains. With the emergence of such powerful models, it has become crucial to effectively leverage their capabilities in tackling challenging vision tasks. On the other hand, only a few works have focused on devising adversarial examples that transfer well to both unknown domains and model architectures. In this paper, we propose a novel transfer attack method called PDCL-Attack, which leverages the CLIP model to enhance the transferability of adversarial perturbations generated by a generative model-based attack framework. Specifically, we formulate an effective prompt-driven feature guidance by harnessing the semantic representation power of text, particularly from the ground-truth class labels of input images. To the best of our knowledge, we are the first to introduce prompt learning to enhance the transferable generative attacks. Extensive experiments conducted across various cross-domain and cross-model settings empirically validate our approach, demonstrating its superiority over state-of-the-art methods.

摘要: 最近的视觉语言基础模型，如CLIP，在学习表征方面表现出了优越的能力，这些表征可以跨不同范围的下游任务和领域转移。随着如此强大的模型的出现，有效地利用它们的能力来处理具有挑战性的愿景任务变得至关重要。另一方面，只有少数工作专注于设计能够很好地移植到未知领域和模型体系结构的对抗性例子。本文提出了一种新的转移攻击方法PDCL-Attack，该方法利用CLIP模型来增强基于产生式模型的攻击框架产生的敌意扰动的可转移性。具体地说，我们通过利用文本的语义表征能力，特别是从输入图像的基本事实类标签来制定有效的提示驱动的特征指导。据我们所知，我们是第一个引入快速学习来增强可转移的生成性攻击的。在各种跨域和跨模型环境下进行的广泛实验验证了我们的方法，证明了它比最先进的方法更优越。



## **20. FACL-Attack: Frequency-Aware Contrastive Learning for Transferable Adversarial Attacks**

FACL攻击：可转移对抗攻击的频率感知对比学习 cs.CV

Accepted to AAAI 2024, Project Page: https://FACL-Attack.github.io

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.20653v1) [paper-pdf](http://arxiv.org/pdf/2407.20653v1)

**Authors**: Hunmin Yang, Jongoh Jeong, Kuk-Jin Yoon

**Abstract**: Deep neural networks are known to be vulnerable to security risks due to the inherent transferable nature of adversarial examples. Despite the success of recent generative model-based attacks demonstrating strong transferability, it still remains a challenge to design an efficient attack strategy in a real-world strict black-box setting, where both the target domain and model architectures are unknown. In this paper, we seek to explore a feature contrastive approach in the frequency domain to generate adversarial examples that are robust in both cross-domain and cross-model settings. With that goal in mind, we propose two modules that are only employed during the training phase: a Frequency-Aware Domain Randomization (FADR) module to randomize domain-variant low- and high-range frequency components and a Frequency-Augmented Contrastive Learning (FACL) module to effectively separate domain-invariant mid-frequency features of clean and perturbed image. We demonstrate strong transferability of our generated adversarial perturbations through extensive cross-domain and cross-model experiments, while keeping the inference time complexity.

摘要: 众所周知，由于对抗性例子的固有可转移性，深度神经网络容易受到安全风险的影响。尽管最近基于模型的生成性攻击取得了成功，表现出很强的可转移性，但在目标域和模型体系结构都未知的现实世界严格的黑盒环境中，设计有效的攻击策略仍然是一个挑战。在本文中，我们试图探索一种在频域中进行特征对比的方法，以生成在跨域和跨模型设置下都具有健壮性的对抗性示例。考虑到这一目标，我们提出了两个仅在训练阶段使用的模块：频率感知域随机化(FADR)模块，用于随机化域变化的低频和高频分量；以及频率增强对比学习(FACL)模块，用于有效分离干净和扰动图像的域不变中频特征。通过广泛的跨域和跨模型实验，我们证明了我们生成的对抗性扰动具有很强的可转移性，同时保持了推理的时间复杂性。



## **21. Robust Federated Learning for Wireless Networks: A Demonstration with Channel Estimation**

无线网络的鲁棒联邦学习：信道估计的演示 cs.LG

Submitted to IEEE GLOBECOM 2024

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2404.03088v2) [paper-pdf](http://arxiv.org/pdf/2404.03088v2)

**Authors**: Zexin Fang, Bin Han, Hans D. Schotten

**Abstract**: Federated learning (FL) offers a privacy-preserving collaborative approach for training models in wireless networks, with channel estimation emerging as a promising application. Despite extensive studies on FL-empowered channel estimation, the security concerns associated with FL require meticulous attention. In a scenario where small base stations (SBSs) serve as local models trained on cached data, and a macro base station (MBS) functions as the global model setting, an attacker can exploit the vulnerability of FL, launching attacks with various adversarial attacks or deployment tactics. In this paper, we analyze such vulnerabilities, corresponding solutions were brought forth, and validated through simulation.

摘要: 联合学习（FL）为无线网络中的训练模型提供了一种保护隐私的协作方法，而信道估计正在成为一个有前途的应用。尽管对FL授权的信道估计进行了广泛的研究，但与FL相关的安全问题需要仔细关注。在小型基站（SBS）充当在缓存数据上训练的本地模型，宏基站（MBS）充当全局模型设置的场景中，攻击者可以利用FL的漏洞，通过各种对抗性攻击或部署策略发起攻击。本文对此类漏洞进行了分析，提出了相应的解决方案，并通过仿真进行了验证。



## **22. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

从ML到LLM：评估网络钓鱼网页检测模型对抗对抗攻击的稳健性 cs.CR

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.20361v1) [paper-pdf](http://arxiv.org/pdf/2407.20361v1)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of two existing models, Stack model and Phishpedia, in classifying PhishOracle-generated adversarial phishing webpages. Additionally, we study a commercial large language model, Gemini Pro Vision, in the context of adversarial attacks. We conduct a user study to determine whether PhishOracle-generated adversarial phishing webpages deceive users. Our findings reveal that many PhishOracle-generated phishing webpages evade current phishing webpage detection models and deceive users, but Gemini Pro Vision is robust to the attack. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources are publicly available on GitHub.

摘要: 网络钓鱼攻击试图欺骗用户窃取敏感信息，构成重大的网络安全威胁。机器学习(ML)和深度学习(DL)的进步导致了许多钓鱼网页检测解决方案的发展，但这些模型仍然容易受到对手攻击。评估它们对敌意网络钓鱼网页的健壮性是至关重要的。现有工具包含为有限数量的品牌预先设计的钓鱼网页的数据集，并且在钓鱼功能方面缺乏多样性。为了应对这些挑战，我们开发了PhishOracle，这是一个通过在合法网页中嵌入不同的钓鱼功能来生成敌意钓鱼网页的工具。我们评估了现有的两种模型Stack模型和Phishpedia模型对PhishOracle生成的敌意钓鱼网页进行分类的稳健性。此外，我们研究了一个商业大型语言模型，Gemini Pro Vision，在对抗性攻击的背景下。我们进行了一项用户研究，以确定PhishOracle生成的敌意钓鱼网页是否欺骗了用户。我们的研究结果显示，许多PhishOracle生成的钓鱼网页逃避了当前的钓鱼网页检测模型并欺骗用户，但Gemini Pro Vision对攻击具有健壮性。我们还开发了PhishOracle Web应用程序，允许用户输入合法的URL，选择相关的网络钓鱼功能并生成相应的网络钓鱼网页。所有资源都在GitHub上公开提供。



## **23. Prompt Leakage effect and defense strategies for multi-turn LLM interactions**

多圈LLM相互作用的即时泄漏效应和防御策略 cs.CR

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2404.16251v3) [paper-pdf](http://arxiv.org/pdf/2404.16251v3)

**Authors**: Divyansh Agarwal, Alexander R. Fabbri, Ben Risher, Philippe Laban, Shafiq Joty, Chien-Sheng Wu

**Abstract**: Prompt leakage poses a compelling security and privacy threat in LLM applications. Leakage of system prompts may compromise intellectual property, and act as adversarial reconnaissance for an attacker. A systematic evaluation of prompt leakage threats and mitigation strategies is lacking, especially for multi-turn LLM interactions. In this paper, we systematically investigate LLM vulnerabilities against prompt leakage for 10 closed- and open-source LLMs, across four domains. We design a unique threat model which leverages the LLM sycophancy effect and elevates the average attack success rate (ASR) from 17.7% to 86.2% in a multi-turn setting. Our standardized setup further allows dissecting leakage of specific prompt contents such as task instructions and knowledge documents. We measure the mitigation effect of 7 black-box defense strategies, along with finetuning an open-source model to defend against leakage attempts. We present different combination of defenses against our threat model, including a cost analysis. Our study highlights key takeaways for building secure LLM applications and provides directions for research in multi-turn LLM interactions

摘要: 即时泄漏在LLM应用程序中构成了令人信服的安全和隐私威胁。泄露系统提示可能会危及知识产权，并充当攻击者的对抗性侦察。缺乏对即时泄漏威胁和缓解策略的系统评估，特别是对于多回合的LLM相互作用。在这篇文章中，我们系统地研究了10个封闭和开放源代码的LLM在四个域中针对即时泄漏的LLM漏洞。我们设计了一种独特的威胁模型，该模型利用了LLM的奉承效应，在多回合环境下将平均攻击成功率从17.7%提高到86.2%。我们的标准化设置还允许剖析特定提示内容的泄漏，如任务说明和知识文档。我们测量了7种黑盒防御策略的缓解效果，并微调了一个开源模型来防御泄漏企图。针对我们的威胁模型，我们提出了不同的防御组合，包括成本分析。我们的研究强调了构建安全的LLM应用程序的关键要点，并为多轮LLM交互的研究提供了方向



## **24. DDAP: Dual-Domain Anti-Personalization against Text-to-Image Diffusion Models**

DDAP：针对文本到图像扩散模型的双域反个性化 cs.CV

Accepted by IJCB 2024

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.20141v1) [paper-pdf](http://arxiv.org/pdf/2407.20141v1)

**Authors**: Jing Yang, Runping Xi, Yingxin Lai, Xun Lin, Zitong Yu

**Abstract**: Diffusion-based personalized visual content generation technologies have achieved significant breakthroughs, allowing for the creation of specific objects by just learning from a few reference photos. However, when misused to fabricate fake news or unsettling content targeting individuals, these technologies could cause considerable societal harm. To address this problem, current methods generate adversarial samples by adversarially maximizing the training loss, thereby disrupting the output of any personalized generation model trained with these samples. However, the existing methods fail to achieve effective defense and maintain stealthiness, as they overlook the intrinsic properties of diffusion models. In this paper, we introduce a novel Dual-Domain Anti-Personalization framework (DDAP). Specifically, we have developed Spatial Perturbation Learning (SPL) by exploiting the fixed and perturbation-sensitive nature of the image encoder in personalized generation. Subsequently, we have designed a Frequency Perturbation Learning (FPL) method that utilizes the characteristics of diffusion models in the frequency domain. The SPL disrupts the overall texture of the generated images, while the FPL focuses on image details. By alternating between these two methods, we construct the DDAP framework, effectively harnessing the strengths of both domains. To further enhance the visual quality of the adversarial samples, we design a localization module to accurately capture attentive areas while ensuring the effectiveness of the attack and avoiding unnecessary disturbances in the background. Extensive experiments on facial benchmarks have shown that the proposed DDAP enhances the disruption of personalized generation models while also maintaining high quality in adversarial samples, making it more effective in protecting privacy in practical applications.

摘要: 基于扩散的个性化视觉内容生成技术取得了重大突破，只需从几张参考照片中学习，就可以创建特定的对象。然而，当这些技术被滥用来编造假新闻或针对个人的令人不安的内容时，可能会造成相当大的社会危害。为了解决这个问题，当前的方法通过对抗性地最大化训练损失来生成对抗性样本，从而扰乱用这些样本训练的任何个性化生成模型的输出。然而，现有的方法忽略了扩散模型的内在特性，无法实现有效的防御和保持隐蔽性。提出了一种新型的双域反个性化框架(DDAP)。具体地说，我们通过在个性化生成中利用图像编码器的固定和对扰动敏感的性质来发展空间扰动学习(SPL)。随后，我们设计了一种利用频域扩散模型特性的频率微扰学习(FPL)方法。SPL破坏了生成图像的整体纹理，而FPL则专注于图像细节。通过在这两种方法之间交替使用，我们构建了DDAP框架，有效地利用了这两个领域的优势。为了进一步提高对手样本的视觉质量，我们设计了定位模块，在保证攻击有效性的同时，准确地捕捉到关注区域，避免了背景中不必要的干扰。在人脸基准上的广泛实验表明，所提出的DDAP增强了个性化生成模型的颠覆性，同时保持了对手样本的高质量，使其在实际应用中更有效地保护隐私。



## **25. Adversarial Robustness in RGB-Skeleton Action Recognition: Leveraging Attention Modality Reweighter**

RGB骨架动作识别中的对抗鲁棒性：利用注意力情态重新加权 cs.CV

Accepted by IJCB 2024

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.19981v1) [paper-pdf](http://arxiv.org/pdf/2407.19981v1)

**Authors**: Chao Liu, Xin Liu, Zitong Yu, Yonghong Hou, Huanjing Yue, Jingyu Yang

**Abstract**: Deep neural networks (DNNs) have been applied in many computer vision tasks and achieved state-of-the-art (SOTA) performance. However, misclassification will occur when DNNs predict adversarial examples which are created by adding human-imperceptible adversarial noise to natural examples. This limits the application of DNN in security-critical fields. In order to enhance the robustness of models, previous research has primarily focused on the unimodal domain, such as image recognition and video understanding. Although multi-modal learning has achieved advanced performance in various tasks, such as action recognition, research on the robustness of RGB-skeleton action recognition models is scarce. In this paper, we systematically investigate how to improve the robustness of RGB-skeleton action recognition models. We initially conducted empirical analysis on the robustness of different modalities and observed that the skeleton modality is more robust than the RGB modality. Motivated by this observation, we propose the \formatword{A}ttention-based \formatword{M}odality \formatword{R}eweighter (\formatword{AMR}), which utilizes an attention layer to re-weight the two modalities, enabling the model to learn more robust features. Our AMR is plug-and-play, allowing easy integration with multimodal models. To demonstrate the effectiveness of AMR, we conducted extensive experiments on various datasets. For example, compared to the SOTA methods, AMR exhibits a 43.77\% improvement against PGD20 attacks on the NTU-RGB+D 60 dataset. Furthermore, it effectively balances the differences in robustness between different modalities.

摘要: 深度神经网络(DNN)已被应用于许多计算机视觉任务中，并取得了最先进的性能(SOTA)。然而，当DNN预测通过在自然实例中添加人类不可察觉的对抗性噪声而产生的对抗性实例时，就会发生误分类。这限制了DNN在安全关键领域的应用。为了增强模型的稳健性，以往的研究主要集中在单峰领域，如图像识别和视频理解。尽管多通道学习在动作识别等各种任务中取得了较好的性能，但对RGB骨架动作识别模型的稳健性研究还很少。本文系统地研究了如何提高RGB-骨架动作识别模型的健壮性。我们最初对不同通道的稳健性进行了实证分析，观察到骨架通道比RGB通道更健壮。基于这一观察结果，我们提出了基于格式词{A}时延的格式词{M}奇异性\格式词{R}权重器(\Formatword{AMR})，它利用关注层对两个通道进行重新加权，使模型能够学习更健壮的特征。我们的AMR是即插即用的，允许与多模式模型轻松集成。为了验证AMR的有效性，我们在不同的数据集上进行了广泛的实验。例如，与SOTA方法相比，AMR在NTU-RGB+D60数据集上对PGD20攻击的性能提高了43.77%。此外，它有效地平衡了不同模式之间在稳健性方面的差异。



## **26. Quasi-Framelets: Robust Graph Neural Networks via Adaptive Framelet Convolution**

准框架集：通过自适应框架集卷积的鲁棒图神经网络 cs.LG

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2201.04728v2) [paper-pdf](http://arxiv.org/pdf/2201.04728v2)

**Authors**: Mengxi Yang, Dai Shi, Xuebin Zheng, Jie Yin, Junbin Gao

**Abstract**: This paper aims to provide a novel design of a multiscale framelet convolution for spectral graph neural networks (GNNs). While current spectral methods excel in various graph learning tasks, they often lack the flexibility to adapt to noisy, incomplete, or perturbed graph signals, making them fragile in such conditions. Our newly proposed framelet convolution addresses these limitations by decomposing graph data into low-pass and high-pass spectra through a finely-tuned multiscale approach. Our approach directly designs filtering functions within the spectral domain, allowing for precise control over the spectral components. The proposed design excels in filtering out unwanted spectral information and significantly reduces the adverse effects of noisy graph signals. Our approach not only enhances the robustness of GNNs but also preserves crucial graph features and structures. Through extensive experiments on diverse, real-world graph datasets, we demonstrate that our framelet convolution achieves superior performance in node classification tasks. It exhibits remarkable resilience to noisy data and adversarial attacks, highlighting its potential as a robust solution for real-world graph applications. This advancement opens new avenues for more adaptive and reliable spectral GNN architectures.

摘要: 本文旨在为谱图神经网络提供一种新的多尺度框架卷积设计。虽然目前的谱方法在各种图形学习任务中表现出色，但它们往往缺乏适应噪声、不完整或扰动的图形信号的灵活性，这使得它们在这些条件下很脆弱。我们最新提出的框架卷积通过精细调整的多尺度方法将图形数据分解成低通和高通光谱，从而解决了这些限制。我们的方法直接在谱域内设计滤波函数，允许对谱分量进行精确控制。所提出的设计在滤除不需要的光谱信息方面表现出色，并显著降低了噪声图形信号的不利影响。我们的方法不仅增强了GNN的健壮性，而且保留了关键的图特征和结构。通过在不同的真实图形数据集上的广泛实验，我们证明了我们的框架集卷积在节点分类任务中取得了优异的性能。它表现出了对噪声数据和对手攻击的非凡弹性，突出了它作为现实世界图形应用程序的健壮解决方案的潜力。这一进展为更适应和更可靠的频谱GNN体系结构开辟了新的途径。



## **27. Detecting and Understanding Vulnerabilities in Language Models via Mechanistic Interpretability**

通过机械解释性检测和理解语言模型中的漏洞 cs.LG

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.19842v1) [paper-pdf](http://arxiv.org/pdf/2407.19842v1)

**Authors**: Jorge García-Carrasco, Alejandro Maté, Juan Trujillo

**Abstract**: Large Language Models (LLMs), characterized by being trained on broad amounts of data in a self-supervised manner, have shown impressive performance across a wide range of tasks. Indeed, their generative abilities have aroused interest on the application of LLMs across a wide range of contexts. However, neural networks in general, and LLMs in particular, are known to be vulnerable to adversarial attacks, where an imperceptible change to the input can mislead the output of the model. This is a serious concern that impedes the use of LLMs on high-stakes applications, such as healthcare, where a wrong prediction can imply serious consequences. Even though there are many efforts on making LLMs more robust to adversarial attacks, there are almost no works that study \emph{how} and \emph{where} these vulnerabilities that make LLMs prone to adversarial attacks happen. Motivated by these facts, we explore how to localize and understand vulnerabilities, and propose a method, based on Mechanistic Interpretability (MI) techniques, to guide this process. Specifically, this method enables us to detect vulnerabilities related to a concrete task by (i) obtaining the subset of the model that is responsible for that task, (ii) generating adversarial samples for that task, and (iii) using MI techniques together with the previous samples to discover and understand the possible vulnerabilities. We showcase our method on a pretrained GPT-2 Small model carrying out the task of predicting 3-letter acronyms to demonstrate its effectiveness on locating and understanding concrete vulnerabilities of the model.

摘要: 大型语言模型(LLM)的特点是以自我监督的方式对大量数据进行培训，在各种任务中表现出令人印象深刻的表现。事实上，他们的生成能力已经引起了人们对LLMS在广泛背景下的应用的兴趣。然而，一般的神经网络，特别是LLM，都很容易受到敌意攻击，在这种攻击中，输入的不可察觉的变化可能会误导模型的输出。这是一个严重的担忧，阻碍了低成本管理在高风险应用中的使用，例如医疗保健，在这些应用中，错误的预测可能意味着严重的后果。尽管已经有许多努力使LLM对对手攻击更健壮，但几乎没有研究这些使LLM容易受到对手攻击的漏洞的著作。在这些事实的推动下，我们探索了如何定位和理解漏洞，并提出了一种基于机械可解释性(MI)技术的方法来指导这一过程。具体地说，这种方法使我们能够通过(I)获得负责该任务的模型的子集，(Ii)为该任务生成对抗性样本，以及(Iii)结合使用MI技术和先前的样本来发现和理解可能的漏洞来检测与该具体任务相关的漏洞。我们在一个预先训练的GPT-2小模型上展示了我们的方法，该模型执行了预测3字母首字母缩写的任务，以演示其在定位和理解模型的具体漏洞方面的有效性。



## **28. Enhancing Adversarial Text Attacks on BERT Models with Projected Gradient Descent**

利用投影梯度下降增强BERT模型的对抗性文本攻击 cs.LG

This paper is the pre-reviewed version of our paper that has been  accepted for oral presentation and publication in the 4th IEEE ASIANCON. The  conference will be organized in Pune, INDIA from August 23 to 25, 2024. The  paper consists of 8 pages and it contains 10 tables. It is NOT the final  camera-ready version that will be in IEEE Xplore

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.21073v1) [paper-pdf](http://arxiv.org/pdf/2407.21073v1)

**Authors**: Hetvi Waghela, Jaydip Sen, Sneha Rakshit

**Abstract**: Adversarial attacks against deep learning models represent a major threat to the security and reliability of natural language processing (NLP) systems. In this paper, we propose a modification to the BERT-Attack framework, integrating Projected Gradient Descent (PGD) to enhance its effectiveness and robustness. The original BERT-Attack, designed for generating adversarial examples against BERT-based models, suffers from limitations such as a fixed perturbation budget and a lack of consideration for semantic similarity. The proposed approach in this work, PGD-BERT-Attack, addresses these limitations by leveraging PGD to iteratively generate adversarial examples while ensuring both imperceptibility and semantic similarity to the original input. Extensive experiments are conducted to evaluate the performance of PGD-BERT-Attack compared to the original BERT-Attack and other baseline methods. The results demonstrate that PGD-BERT-Attack achieves higher success rates in causing misclassification while maintaining low perceptual changes. Furthermore, PGD-BERT-Attack produces adversarial instances that exhibit greater semantic resemblance to the initial input, enhancing their applicability in real-world scenarios. Overall, the proposed modification offers a more effective and robust approach to adversarial attacks on BERT-based models, thus contributing to the advancement of defense against attacks on NLP systems.

摘要: 针对深度学习模型的敌意攻击是对自然语言处理(NLP)系统安全性和可靠性的主要威胁。在本文中，我们提出了一种改进的BERT攻击框架，结合投影梯度下降(PGD)来增强其有效性和稳健性。最初的BERT攻击旨在生成针对基于BERT的模型的敌意示例，但存在诸如固定的扰动预算和缺乏对语义相似性的考虑等限制。本文提出的方法PGD-BERT-Attack，通过利用PGD迭代生成对抗性实例，同时确保不可感知性和与原始输入的语义相似性，解决了这些限制。通过大量的实验，评估了PGD-BERT-攻击与原始BERT-攻击和其他基线方法的性能。实验结果表明，PGD-BERT-Attack在保持较低感知变化的情况下，具有较高的误分类成功率。此外，PGD-BERT-Attack生成的对抗性实例与初始输入具有更大的语义相似性，从而增强了它们在真实场景中的适用性。总体而言，拟议的修改提供了一种更有效和更健壮的方法来应对基于BERT的模型的对抗性攻击，从而有助于提高对NLP系统攻击的防御。



## **29. Understanding Robust Overfitting from the Feature Generalization Perspective**

从特征概括的角度理解稳健过拟 cs.LG

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2310.00607v2) [paper-pdf](http://arxiv.org/pdf/2310.00607v2)

**Authors**: Chaojian Yu, Xiaolong Shi, Jun Yu, Bo Han, Tongliang Liu

**Abstract**: Adversarial training (AT) constructs robust neural networks by incorporating adversarial perturbations into natural data. However, it is plagued by the issue of robust overfitting (RO), which severely damages the model's robustness. In this paper, we investigate RO from a novel feature generalization perspective. Specifically, we design factor ablation experiments to assess the respective impacts of natural data and adversarial perturbations on RO, identifying that the inducing factor of RO stems from natural data. Given that the only difference between adversarial and natural training lies in the inclusion of adversarial perturbations, we further hypothesize that adversarial perturbations degrade the generalization of features in natural data and verify this hypothesis through extensive experiments. Based on these findings, we provide a holistic view of RO from the feature generalization perspective and explain various empirical behaviors associated with RO. To examine our feature generalization perspective, we devise two representative methods, attack strength and data augmentation, to prevent the feature generalization degradation during AT. Extensive experiments conducted on benchmark datasets demonstrate that the proposed methods can effectively mitigate RO and enhance adversarial robustness.

摘要: 对抗性训练(AT)通过在自然数据中加入对抗性扰动来构建稳健的神经网络。然而，稳健过拟合(RO)问题严重影响了模型的稳健性。在本文中，我们从一种新的特征泛化的角度来研究RO。具体地说，我们设计了因子消融实验来评估自然数据和对抗性扰动对RO的影响，确定了RO的诱导因素源于自然数据。鉴于对抗性训练和自然训练之间的唯一区别在于包含对抗性扰动，我们进一步假设对抗性扰动降低了自然数据中特征的泛化，并通过广泛的实验验证了这一假设。基于这些发现，我们从特征泛化的角度提供了关于反语的整体观点，并解释了与反语相关的各种经验行为。为了检验我们的特征泛化观点，我们设计了两种有代表性的方法，攻击强度和数据增强，以防止特征泛化在AT过程中的退化。在基准数据集上进行的大量实验表明，所提出的方法可以有效地缓解RO，增强对手的健壮性。



## **30. Exploring the Adversarial Robustness of CLIP for AI-generated Image Detection**

探索CLIP用于人工智能生成图像检测的对抗鲁棒性 cs.CV

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2407.19553v1) [paper-pdf](http://arxiv.org/pdf/2407.19553v1)

**Authors**: Vincenzo De Rosa, Fabrizio Guillaro, Giovanni Poggi, Davide Cozzolino, Luisa Verdoliva

**Abstract**: In recent years, many forensic detectors have been proposed to detect AI-generated images and prevent their use for malicious purposes. Convolutional neural networks (CNNs) have long been the dominant architecture in this field and have been the subject of intense study. However, recently proposed Transformer-based detectors have been shown to match or even outperform CNN-based detectors, especially in terms of generalization. In this paper, we study the adversarial robustness of AI-generated image detectors, focusing on Contrastive Language-Image Pretraining (CLIP)-based methods that rely on Visual Transformer backbones and comparing their performance with CNN-based methods. We study the robustness to different adversarial attacks under a variety of conditions and analyze both numerical results and frequency-domain patterns. CLIP-based detectors are found to be vulnerable to white-box attacks just like CNN-based detectors. However, attacks do not easily transfer between CNN-based and CLIP-based methods. This is also confirmed by the different distribution of the adversarial noise patterns in the frequency domain. Overall, this analysis provides new insights into the properties of forensic detectors that can help to develop more effective strategies.

摘要: 近年来，已经提出了许多法医检测器来检测人工智能生成的图像，并防止将其用于恶意目的。卷积神经网络(CNN)长期以来一直是这一领域的主导结构，也一直是研究的热点。然而，最近提出的基于变形金刚的检测器已经被证明与基于CNN的检测器相媲美，甚至优于基于CNN的检测器，特别是在泛化方面。在本文中，我们研究了人工智能生成的图像检测器的对抗鲁棒性，重点研究了基于对比语言图像预训练(CLIP)的依赖于视觉变形金刚的方法，并将它们的性能与基于CNN的方法进行了比较。我们研究了在不同条件下对不同敌意攻击的鲁棒性，并对数值结果和频域模式进行了分析。基于剪辑的检测器被发现像基于CNN的检测器一样容易受到白盒攻击。然而，攻击并不容易在基于CNN的方法和基于剪辑的方法之间转移。对抗性噪声模式在频域中的不同分布也证实了这一点。总体而言，这一分析为法医探测器的特性提供了新的见解，有助于制定更有效的战略。



## **31. Learning on Graphs with Large Language Models(LLMs): A Deep Dive into Model Robustness**

使用大型语言模型（LLM）学习图形：深入研究模型稳健性 cs.LG

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2407.12068v2) [paper-pdf](http://arxiv.org/pdf/2407.12068v2)

**Authors**: Kai Guo, Zewen Liu, Zhikai Chen, Hongzhi Wen, Wei Jin, Jiliang Tang, Yi Chang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing tasks. Recently, several LLMs-based pipelines have been developed to enhance learning on graphs with text attributes, showcasing promising performance. However, graphs are well-known to be susceptible to adversarial attacks and it remains unclear whether LLMs exhibit robustness in learning on graphs. To address this gap, our work aims to explore the potential of LLMs in the context of adversarial attacks on graphs. Specifically, we investigate the robustness against graph structural and textual perturbations in terms of two dimensions: LLMs-as-Enhancers and LLMs-as-Predictors. Through extensive experiments, we find that, compared to shallow models, both LLMs-as-Enhancers and LLMs-as-Predictors offer superior robustness against structural and textual attacks.Based on these findings, we carried out additional analyses to investigate the underlying causes. Furthermore, we have made our benchmark library openly available to facilitate quick and fair evaluations, and to encourage ongoing innovative research in this field.

摘要: 大型语言模型(LLM)在各种自然语言处理任务中表现出了显著的性能。最近，已经开发了几个基于LLMS的管道来增强对具有文本属性的图形的学习，展示了良好的性能。然而，众所周知，图是容易受到敌意攻击的，而且目前还不清楚LLM是否表现出关于图的学习的健壮性。为了解决这一差距，我们的工作旨在探索LLMS在对抗性攻击图的背景下的潜力。具体地说，我们从两个维度考察了对图结构和文本扰动的稳健性：作为增强器的LLMS和作为预测者的LLMS。通过广泛的实验，我们发现，与浅层模型相比，LLMS-as-Enhancer和LLMS-as-Predicator对结构和文本攻击都具有更好的稳健性。基于这些发现，我们进行了额外的分析，以探讨潜在的原因。此外，我们开放了我们的基准图书馆，以促进快速和公平的评估，并鼓励这一领域正在进行的创新研究。



## **32. Breaking the Balance of Power: Commitment Attacks on Ethereum's Reward Mechanism**

打破权力平衡：承诺攻击以太坊奖励机制 cs.CR

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2407.19479v1) [paper-pdf](http://arxiv.org/pdf/2407.19479v1)

**Authors**: Roozbeh Sarenche, Ertem Nusret Tas, Barnabe Monnot, Caspar Schwarz-Schilling, Bart Preneel

**Abstract**: Validators in permissionless, large-scale blockchains (e.g., Ethereum) are typically payoff-maximizing, rational actors. Ethereum relies on in-protocol incentives, like rewards for validators delivering correct and timely votes, to induce honest behavior and secure the blockchain. However, external incentives, such as the block proposer's opportunity to capture maximal extractable value (MEV), may tempt validators to deviate from honest protocol participation.   We show a series of commitment attacks on LMD GHOST, a core part of Ethereum's consensus mechanism. We demonstrate how a single adversarial block proposer can orchestrate long-range chain reorganizations by manipulating Ethereum's reward system for timely votes. These attacks disrupt the intended balance of power between proposers and voters: by leveraging credible threats, the adversarial proposer can coerce voters from previous slots into supporting blocks that conflict with the honest chain, enabling a chain reorganization at no cost to the adversary. In response, we introduce a novel reward mechanism that restores the voters' role as a check against proposer power. Our proposed mitigation is fairer and more decentralized -- not only in the context of these attacks -- but also practical for implementation in Ethereum.

摘要: 未经许可的大规模区块链(例如，以太)中的验证器通常是收益最大化的理性参与者。Etherum依赖于协议内激励，例如对提供正确和及时投票的验证者的奖励，以诱导诚实的行为并保护区块链。然而，外部激励，如区块提出者获取最大可提取价值(MEV)的机会，可能会诱使验证者偏离诚实的协议参与。我们展示了对LMD Ghost的一系列承诺攻击，LMD Ghost是以太共识机制的核心部分。我们展示了一个单一的对抗性区块提出者如何通过操纵以太的奖励系统来协调远程链重组，以获得及时的投票。这些攻击破坏了提倡者和选民之间预期的权力平衡：通过利用可信的威胁，对抗性的提倡者可以迫使选民从之前的位置进入与诚实链冲突的支持块，从而在不对对手造成任何成本的情况下实现链重组。作为回应，我们引入了一种新颖的奖励机制，恢复了选民对提议权力的制衡作用。我们提议的缓解措施更公平和更分散--不仅在这些攻击的背景下--而且在以太实施方面也是可行的。



## **33. Predominant Aspects on Security for Quantum Machine Learning: Literature Review**

量子机器学习安全性的主要方面：文献评论 quant-ph

Accepted at the IEEE International Conference on Quantum Computing  and Engineering (QCE)

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2401.07774v3) [paper-pdf](http://arxiv.org/pdf/2401.07774v3)

**Authors**: Nicola Franco, Alona Sakhnenko, Leon Stolpmann, Daniel Thuerck, Fabian Petsch, Annika Rüll, Jeanette Miriam Lorenz

**Abstract**: Quantum Machine Learning (QML) has emerged as a promising intersection of quantum computing and classical machine learning, anticipated to drive breakthroughs in computational tasks. This paper discusses the question which security concerns and strengths are connected to QML by means of a systematic literature review. We categorize and review the security of QML models, their vulnerabilities inherent to quantum architectures, and the mitigation strategies proposed. The survey reveals that while QML possesses unique strengths, it also introduces novel attack vectors not seen in classical systems. We point out specific risks, such as cross-talk in superconducting systems and forced repeated shuttle operations in ion-trap systems, which threaten QML's reliability. However, approaches like adversarial training, quantum noise exploitation, and quantum differential privacy have shown potential in enhancing QML robustness. Our review discuss the need for continued and rigorous research to ensure the secure deployment of QML in real-world applications. This work serves as a foundational reference for researchers and practitioners aiming to navigate the security aspects of QML.

摘要: 量子机器学习(QML)已经成为量子计算和经典机器学习的一个有前途的交叉点，有望推动计算任务的突破。本文通过系统的文献综述，探讨了QML的安全关注点和优势所在。我们对QML模型的安全性、量子体系结构固有的脆弱性以及提出的缓解策略进行了分类和回顾。调查显示，虽然QML具有独特的优势，但它也引入了经典系统中未曾见过的新攻击载体。我们指出了具体的风险，如超导系统中的串扰和离子陷阱系统中被迫重复的穿梭操作，这些风险威胁到了QML的可靠性。然而，对抗性训练、量子噪声利用和量子差分隐私等方法已经显示出在增强QML稳健性方面的潜力。我们的综述讨论了持续和严格研究的必要性，以确保QML在现实世界应用程序中的安全部署。这项工作为旨在导航QML安全方面的研究人员和实践者提供了基础性参考。



## **34. A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks**

仔细研究GAN先验：利用中间功能进行增强模型倒置攻击 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.13863v3) [paper-pdf](http://arxiv.org/pdf/2407.13863v3)

**Authors**: Yixiang Qiu, Hao Fang, Hongyao Yu, Bin Chen, MeiKang Qiu, Shu-Tao Xia

**Abstract**: Model Inversion (MI) attacks aim to reconstruct privacy-sensitive training data from released models by utilizing output information, raising extensive concerns about the security of Deep Neural Networks (DNNs). Recent advances in generative adversarial networks (GANs) have contributed significantly to the improved performance of MI attacks due to their powerful ability to generate realistic images with high fidelity and appropriate semantics. However, previous MI attacks have solely disclosed private information in the latent space of GAN priors, limiting their semantic extraction and transferability across multiple target models and datasets. To address this challenge, we propose a novel method, Intermediate Features enhanced Generative Model Inversion (IF-GMI), which disassembles the GAN structure and exploits features between intermediate blocks. This allows us to extend the optimization space from latent code to intermediate features with enhanced expressive capabilities. To prevent GAN priors from generating unrealistic images, we apply a L1 ball constraint to the optimization process. Experiments on multiple benchmarks demonstrate that our method significantly outperforms previous approaches and achieves state-of-the-art results under various settings, especially in the out-of-distribution (OOD) scenario. Our code is available at: https://github.com/final-solution/IF-GMI

摘要: 模型反转(MI)攻击的目的是利用输出信息从已发布的模型中重建隐私敏感的训练数据，这引起了人们对深度神经网络(DNN)安全性的广泛关注。生成性对抗网络(GANS)的最新进展为MI攻击的性能改进做出了重要贡献，因为它们能够生成高保真和适当语义的真实图像。然而，以往的MI攻击只在GaN先验的潜在空间中泄露隐私信息，限制了它们的语义提取和跨多个目标模型和数据集的可传输性。为了解决这一挑战，我们提出了一种新的方法，中间特征增强的生成性模型反转(IF-GMI)，它分解GaN结构并利用中间块之间的特征。这允许我们将优化空间从潜在代码扩展到具有增强表达能力的中间功能。为了防止GaN先验数据产生不真实的图像，我们在优化过程中应用了L1球约束。在多个基准测试上的实验表明，我们的方法显著优于以前的方法，并在各种设置下获得了最先进的结果，特别是在分布外(OOD)的情况下。我们的代码请访问：https://github.com/final-solution/IF-GMI



## **35. EaTVul: ChatGPT-based Evasion Attack Against Software Vulnerability Detection**

EaTVul：针对软件漏洞检测的基于ChatGPT的规避攻击 cs.CR

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.19216v1) [paper-pdf](http://arxiv.org/pdf/2407.19216v1)

**Authors**: Shigang Liu, Di Cao, Junae Kim, Tamas Abraham, Paul Montague, Seyit Camtepe, Jun Zhang, Yang Xiang

**Abstract**: Recently, deep learning has demonstrated promising results in enhancing the accuracy of vulnerability detection and identifying vulnerabilities in software. However, these techniques are still vulnerable to attacks. Adversarial examples can exploit vulnerabilities within deep neural networks, posing a significant threat to system security. This study showcases the susceptibility of deep learning models to adversarial attacks, which can achieve 100% attack success rate (refer to Table 5). The proposed method, EaTVul, encompasses six stages: identification of important samples using support vector machines, identification of important features using the attention mechanism, generation of adversarial data based on these features using ChatGPT, preparation of an adversarial attack pool, selection of seed data using a fuzzy genetic algorithm, and the execution of an evasion attack. Extensive experiments demonstrate the effectiveness of EaTVul, achieving an attack success rate of more than 83% when the snippet size is greater than 2. Furthermore, in most cases with a snippet size of 4, EaTVul achieves a 100% attack success rate. The findings of this research emphasize the necessity of robust defenses against adversarial attacks in software vulnerability detection.

摘要: 最近，深度学习在提高漏洞检测的准确性和识别软件中的漏洞方面显示了良好的结果。然而，这些技术仍然容易受到攻击。敌意的例子可以利用深层神经网络中的漏洞，对系统安全构成重大威胁。这项研究展示了深度学习模型对对抗性攻击的敏感性，它可以达到100%的攻击成功率(参见表5)。该方法EaTVul包括六个阶段：使用支持向量机识别重要样本，使用注意机制识别重要特征，使用ChatGPT基于这些特征生成对抗性数据，准备对抗性攻击池，使用模糊遗传算法选择种子数据，以及执行规避攻击。大量实验证明了EaTVul的有效性，当Snipment大小大于2时，攻击成功率达到83%以上。此外，在大多数情况下，Snipment大小为4的情况下，EaTVul达到100%的攻击成功率。这项研究的结果强调了在软件漏洞检测中对对手攻击进行稳健防御的必要性。



## **36. Debiased Graph Poisoning Attack via Contrastive Surrogate Objective**

通过对比替代目标的去偏图中毒攻击 cs.LG

9 pages. Proceeding ACM International Conference on Information and  Knowledge Management (CIKM 2024) Proceeding

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.19155v1) [paper-pdf](http://arxiv.org/pdf/2407.19155v1)

**Authors**: Kanghoon Yoon, Yeonjun In, Namkyeong Lee, Kibum Kim, Chanyoung Park

**Abstract**: Graph neural networks (GNN) are vulnerable to adversarial attacks, which aim to degrade the performance of GNNs through imperceptible changes on the graph. However, we find that in fact the prevalent meta-gradient-based attacks, which utilizes the gradient of the loss w.r.t the adjacency matrix, are biased towards training nodes. That is, their meta-gradient is determined by a training procedure of the surrogate model, which is solely trained on the training nodes. This bias manifests as an uneven perturbation, connecting two nodes when at least one of them is a labeled node, i.e., training node, while it is unlikely to connect two unlabeled nodes. However, these biased attack approaches are sub-optimal as they do not consider flipping edges between two unlabeled nodes at all. This means that they miss the potential attacked edges between unlabeled nodes that significantly alter the representation of a node. In this paper, we investigate the meta-gradients to uncover the root cause of the uneven perturbations of existing attacks. Based on our analysis, we propose a Meta-gradient-based attack method using contrastive surrogate objective (Metacon), which alleviates the bias in meta-gradient using a new surrogate loss. We conduct extensive experiments to show that Metacon outperforms existing meta gradient-based attack methods through benchmark datasets, while showing that alleviating the bias towards training nodes is effective in attacking the graph structure.

摘要: 图神经网络(GNN)容易受到敌意攻击，其目的是通过图上不可察觉的变化来降低GNN的性能。然而，我们发现，实际上流行的基于亚梯度的攻击利用邻接矩阵的损失的梯度，偏向于训练节点。也就是说，它们的亚梯度由仅在训练节点上训练的代理模型的训练过程确定。这种偏差表现为不均匀的扰动，当两个节点中至少有一个是已标记节点(即训练节点)时连接两个节点，而不太可能连接两个未标记节点。然而，这些有偏见的攻击方法是次优的，因为它们根本不考虑两个未标记节点之间的翻转边。这意味着它们错过了未标记节点之间的潜在攻击边，这些边显著改变了节点的表示。在本文中，我们研究了亚梯度，以揭示现有攻击的不均匀扰动的根本原因。在此基础上，提出了一种基于元梯度的对比代理目标攻击方法(Metacon)，该方法通过引入新的代理损失来缓解元梯度的偏向。我们通过大量的实验表明，Metacon的攻击性能优于现有的基于元梯度的攻击方法，同时也证明了减轻对训练节点的偏向对图结构的攻击是有效的。



## **37. A Survey of Malware Detection Using Deep Learning**

使用深度学习的恶意软件检测概览 cs.CR

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.19153v1) [paper-pdf](http://arxiv.org/pdf/2407.19153v1)

**Authors**: Ahmed Bensaoud, Jugal Kalita, Mahmoud Bensaoud

**Abstract**: The problem of malicious software (malware) detection and classification is a complex task, and there is no perfect approach. There is still a lot of work to be done. Unlike most other research areas, standard benchmarks are difficult to find for malware detection. This paper aims to investigate recent advances in malware detection on MacOS, Windows, iOS, Android, and Linux using deep learning (DL) by investigating DL in text and image classification, the use of pre-trained and multi-task learning models for malware detection approaches to obtain high accuracy and which the best approach if we have a standard benchmark dataset. We discuss the issues and the challenges in malware detection using DL classifiers by reviewing the effectiveness of these DL classifiers and their inability to explain their decisions and actions to DL developers presenting the need to use Explainable Machine Learning (XAI) or Interpretable Machine Learning (IML) programs. Additionally, we discuss the impact of adversarial attacks on deep learning models, negatively affecting their generalization capabilities and resulting in poor performance on unseen data. We believe there is a need to train and test the effectiveness and efficiency of the current state-of-the-art deep learning models on different malware datasets. We examine eight popular DL approaches on various datasets. This survey will help researchers develop a general understanding of malware recognition using deep learning.

摘要: 恶意软件(Malware)的检测和分类问题是一项复杂的任务，目前还没有完美的方法。还有很多工作要做。与大多数其他研究领域不同，很难找到用于检测恶意软件的标准基准。本文旨在研究深度学习在MacOS、Windows、iOS、Android和Linux上的恶意软件检测方面的最新进展，通过研究深度学习在文本和图像分类中的应用，使用预训练和多任务学习模型来获得高精度的恶意软件检测方法，以及如果我们有标准的基准数据集，哪种方法是最好的方法。我们讨论了使用DL分类器检测恶意软件的问题和挑战，方法是回顾这些DL分类器的有效性，以及它们无法向DL开发人员解释它们的决策和操作，并提出需要使用可解释机器学习(XAI)或可解释机器学习(IML)程序。此外，我们还讨论了敌意攻击对深度学习模型的影响，对其泛化能力产生了负面影响，并导致在不可见数据上的性能较差。我们认为有必要对当前最先进的深度学习模型在不同恶意软件数据集上的有效性和效率进行培训和测试。我们在不同的数据集上检查了八种流行的数据挖掘方法。这项调查将帮助研究人员利用深度学习对恶意软件识别有一个大致的理解。



## **38. Accurate and Scalable Detection and Investigation of Cyber Persistence Threats**

对网络持久性威胁进行准确且可扩展的检测和调查 cs.CR

16 pages

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18832v1) [paper-pdf](http://arxiv.org/pdf/2407.18832v1)

**Authors**: Qi Liu, Muhammad Shoaib, Mati Ur Rehman, Kaibin Bao, Veit Hagenmeyer, Wajih Ul Hassan

**Abstract**: In Advanced Persistent Threat (APT) attacks, achieving stealthy persistence within target systems is often crucial for an attacker's success. This persistence allows adversaries to maintain prolonged access, often evading detection mechanisms. Recognizing its pivotal role in the APT lifecycle, this paper introduces Cyber Persistence Detector (CPD), a novel system dedicated to detecting cyber persistence through provenance analytics. CPD is founded on the insight that persistent operations typically manifest in two phases: the "persistence setup" and the subsequent "persistence execution". By causally relating these phases, we enhance our ability to detect persistent threats. First, CPD discerns setups signaling an impending persistent threat and then traces processes linked to remote connections to identify persistence execution activities. A key feature of our system is the introduction of pseudo-dependency edges (pseudo-edges), which effectively connect these disjoint phases using data provenance analysis, and expert-guided edges, which enable faster tracing and reduced log size. These edges empower us to detect persistence threats accurately and efficiently. Moreover, we propose a novel alert triage algorithm that further reduces false positives associated with persistence threats. Evaluations conducted on well-known datasets demonstrate that our system reduces the average false positive rate by 93% compared to state-of-the-art methods.

摘要: 在高级持久威胁(APT)攻击中，在目标系统内实现隐形持久性通常是攻击者成功的关键。这种持久性使攻击者能够保持长时间的访问，通常可以避开检测机制。认识到其在APT生命周期中的关键作用，本文引入了网络持久性检测器(CPD)，这是一个致力于通过来源分析来检测网络持久性的新系统。CPD建立在持久化操作通常表现为两个阶段的洞察力之上：“持久化设置”和随后的“持久化执行”。通过将这些阶段因果联系起来，我们增强了检测持续威胁的能力。首先，CPD识别发出即将到来的持久威胁的设置，然后跟踪链接到远程连接的进程，以识别持久执行活动。我们系统的一个关键特征是引入了伪依赖边(伪边)，它使用数据来源分析有效地连接了这些不相交的阶段，以及专家指导的边，从而实现了更快的跟踪和减少日志大小。这些边缘使我们能够准确高效地检测持久性威胁。此外，我们还提出了一种新的警报分类算法，进一步减少了与持久性威胁相关的误报。在知名数据集上进行的评估表明，与最先进的方法相比，我们的系统平均假阳性率降低了93%。



## **39. Frosty: Bringing strong liveness guarantees to the Snow family of consensus protocols**

Frosty：为Snow家族的共识协议带来强大的活力保证 cs.DC

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2404.14250v4) [paper-pdf](http://arxiv.org/pdf/2404.14250v4)

**Authors**: Aaron Buchwald, Stephen Buttolph, Andrew Lewis-Pye, Patrick O'Grady, Kevin Sekniqi

**Abstract**: Snowman is the consensus protocol implemented by the Avalanche blockchain and is part of the Snow family of protocols, first introduced through the original Avalanche leaderless consensus protocol. A major advantage of Snowman is that each consensus decision only requires an expected constant communication overhead per processor in the `common' case that the protocol is not under substantial Byzantine attack, i.e. it provides a solution to the scalability problem which ensures that the expected communication overhead per processor is independent of the total number of processors $n$ during normal operation. This is the key property that would enable a consensus protocol to scale to 10,000 or more independent validators (i.e. processors). On the other hand, the two following concerns have remained:   (1) Providing formal proofs of consistency for Snowman has presented a formidable challenge.   (2) Liveness attacks exist in the case that a Byzantine adversary controls more than $O(\sqrt{n})$ processors, slowing termination to more than a logarithmic number of steps.   In this paper, we address the two issues above. We consider a Byzantine adversary that controls at most $f<n/5$ processors. First, we provide a simple proof of consistency for Snowman. Then we supplement Snowman with a `liveness module' that can be triggered in the case that a substantial adversary launches a liveness attack, and which guarantees liveness in this event by temporarily forgoing the communication complexity advantages of Snowman, but without sacrificing these low communication complexity advantages during normal operation.

摘要: 雪人是雪崩区块链实施的共识协议，是雪诺协议家族的一部分，最初是通过最初的雪崩无领导共识协议引入的。Snowman的一个主要优势是，在协议没有受到实质性拜占庭攻击的情况下，每个协商一致的决定只需要每个处理器预期的恒定通信开销，即它提供了对可伸缩性问题的解决方案，该解决方案确保在正常操作期间每个处理器的预期通信开销与处理器总数$n$无关。这是使共识协议能够扩展到10,000个或更多独立验证器(即处理器)的关键属性。另一方面，以下两个问题仍然存在：(1)为雪人提供一致性的正式证据是一个巨大的挑战。(2)当拜占庭敌手控制超过$O(\Sqrt{n})$个处理器时，存在活性攻击，从而将终止速度减慢到超过对数步数。在本文中，我们解决了上述两个问题。我们考虑一个拜占庭对手，它至多控制$f<n/5$处理器。首先，我们为雪人提供了一个简单的一致性证明。然后，我们给Snowman增加了一个活跃度模块，该模块可以在强大的对手发起活跃度攻击的情况下触发，并通过暂时放弃Snowman的通信复杂性优势来保证在这种情况下的活跃性，但在正常运行时不会牺牲这些低通信复杂性的优势。



## **40. Embodied Laser Attack:Leveraging Scene Priors to Achieve Agent-based Robust Non-contact Attacks**

准激光攻击：利用场景先验实现基于代理的鲁棒非接触式攻击 cs.CV

9 pages, 7 figures, Accepted by ACM MM 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2312.09554v3) [paper-pdf](http://arxiv.org/pdf/2312.09554v3)

**Authors**: Yitong Sun, Yao Huang, Xingxing Wei

**Abstract**: As physical adversarial attacks become extensively applied in unearthing the potential risk of security-critical scenarios, especially in dynamic scenarios, their vulnerability to environmental variations has also been brought to light. The non-robust nature of physical adversarial attack methods brings less-than-stable performance consequently. Although methods such as EOT have enhanced the robustness of traditional contact attacks like adversarial patches, they fall short in practicality and concealment within dynamic environments such as traffic scenarios. Meanwhile, non-contact laser attacks, while offering enhanced adaptability, face constraints due to a limited optimization space for their attributes, rendering EOT less effective. This limitation underscores the necessity for developing a new strategy to augment the robustness of such practices. To address these issues, this paper introduces the Embodied Laser Attack (ELA), a novel framework that leverages the embodied intelligence paradigm of Perception-Decision-Control to dynamically tailor non-contact laser attacks. For the perception module, given the challenge of simulating the victim's view by full-image transformation, ELA has innovatively developed a local perspective transformation network, based on the intrinsic prior knowledge of traffic scenes and enables effective and efficient estimation. For the decision and control module, ELA trains an attack agent with data-driven reinforcement learning instead of adopting time-consuming heuristic algorithms, making it capable of instantaneously determining a valid attack strategy with the perceived information by well-designed rewards, which is then conducted by a controllable laser emitter. Experimentally, we apply our framework to diverse traffic scenarios both in the digital and physical world, verifying the effectiveness of our method under dynamic successive scenes.

摘要: 随着物理对抗性攻击被广泛应用于挖掘安全关键场景的潜在风险，特别是在动态场景中，它们对环境变化的脆弱性也暴露了出来。因此，物理对抗性攻击方法的非健壮性带来了不稳定的性能。虽然EOT等方法增强了传统接触攻击(如对抗性补丁)的健壮性，但在交通场景等动态环境中缺乏实用性和隐蔽性。同时，非接触式激光攻击虽然提供了增强的适应性，但由于其属性的优化空间有限而面临约束，使得EOT的效率较低。这一局限性突出表明，有必要制定一项新战略，以加强这类做法的稳健性。为了解决这些问题，本文引入了嵌入式激光攻击(ELA)，这是一种利用感知-决策-控制的嵌入式智能范式来动态定制非接触式激光攻击的框架。对于感知模块，针对通过全图像变换模拟受害者视角的挑战，ELA基于交通场景的固有先验知识，创新性地开发了局部透视变换网络，并实现了有效和高效的估计。对于决策与控制模块，ELA用数据驱动的强化学习来训练攻击代理，而不是采用耗时的启发式算法，使其能够通过精心设计的奖励，根据感知的信息即时确定有效的攻击策略，然后由可控的激光发射器进行。实验中，我们将我们的框架应用于数字和物理世界中不同的交通场景，验证了我们的方法在动态连续场景下的有效性。



## **41. Adversarial Robustification via Text-to-Image Diffusion Models**

通过文本到图像扩散模型的对抗性Robusification cs.CV

Code is available at https://github.com/ChoiDae1/robustify-T2I

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18658v1) [paper-pdf](http://arxiv.org/pdf/2407.18658v1)

**Authors**: Daewon Choi, Jongheon Jeong, Huiwon Jang, Jinwoo Shin

**Abstract**: Adversarial robustness has been conventionally believed as a challenging property to encode for neural networks, requiring plenty of training data. In the recent paradigm of adopting off-the-shelf models, however, access to their training data is often infeasible or not practical, while most of such models are not originally trained concerning adversarial robustness. In this paper, we develop a scalable and model-agnostic solution to achieve adversarial robustness without using any data. Our intuition is to view recent text-to-image diffusion models as "adaptable" denoisers that can be optimized to specify target tasks. Based on this, we propose: (a) to initiate a denoise-and-classify pipeline that offers provable guarantees against adversarial attacks, and (b) to leverage a few synthetic reference images generated from the text-to-image model that enables novel adaptation schemes. Our experiments show that our data-free scheme applied to the pre-trained CLIP could improve the (provable) adversarial robustness of its diverse zero-shot classification derivatives (while maintaining their accuracy), significantly surpassing prior approaches that utilize the full training data. Not only for CLIP, we also demonstrate that our framework is easily applicable for robustifying other visual classifiers efficiently.

摘要: 传统上，对抗健壮性被认为是神经网络编码的一种具有挑战性的特性，需要大量的训练数据。然而，在最近采用现成模型的范例中，获取其训练数据往往是不可行或不实用的，而大多数这种模型最初并没有接受过关于对手稳健性的培训。在本文中，我们开发了一个可扩展的与模型无关的解决方案，以在不使用任何数据的情况下实现对手健壮性。我们的直觉是将最近的文本到图像扩散模型视为可以优化以指定目标任务的“适应性”消噪器。基于此，我们建议：(A)启动去噪和分类管道，提供针对对手攻击的可证明的保证，以及(B)利用从文本到图像模型生成的几个合成参考图像，从而实现新的适应方案。我们的实验表明，我们的无数据方案应用于预先训练的剪辑，可以提高其各种零射击分类导数的(可证明的)对抗健壮性(同时保持其准确性)，显著超过以往利用全部训练数据的方法。不仅对于CLIP，我们还证明了我们的框架可以很容易地适用于高效地增强其他视觉分类器的鲁棒性。



## **42. Robust VAEs via Generating Process of Noise Augmented Data**

通过生成噪音增强数据的过程实现稳健的VAE cs.LG

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18632v1) [paper-pdf](http://arxiv.org/pdf/2407.18632v1)

**Authors**: Hiroo Irobe, Wataru Aoki, Kimihiro Yamazaki, Yuhui Zhang, Takumi Nakagawa, Hiroki Waida, Yuichiro Wada, Takafumi Kanamori

**Abstract**: Advancing defensive mechanisms against adversarial attacks in generative models is a critical research topic in machine learning. Our study focuses on a specific type of generative models - Variational Auto-Encoders (VAEs). Contrary to common beliefs and existing literature which suggest that noise injection towards training data can make models more robust, our preliminary experiments revealed that naive usage of noise augmentation technique did not substantially improve VAE robustness. In fact, it even degraded the quality of learned representations, making VAEs more susceptible to adversarial perturbations. This paper introduces a novel framework that enhances robustness by regularizing the latent space divergence between original and noise-augmented data. Through incorporating a paired probabilistic prior into the standard variational lower bound, our method significantly boosts defense against adversarial attacks. Our empirical evaluations demonstrate that this approach, termed Robust Augmented Variational Auto-ENcoder (RAVEN), yields superior performance in resisting adversarial inputs on widely-recognized benchmark datasets.

摘要: 在产生式模型中提出对抗攻击的防御机制是机器学习中的一个重要研究课题。我们的研究集中在一种特定类型的产生式模型-变分自动编码器(VAE)。与通常的信念和现有文献表明向训练数据注入噪声可以使模型更稳健相反，我们的初步实验表明，幼稚地使用噪声增强技术并没有显著提高VAE的稳健性。事实上，它甚至降低了学习表征的质量，使VAE更容易受到对抗性干扰。本文介绍了一种新的框架，它通过正则化原始数据和噪声增强数据之间的潜在空间发散来增强稳健性。通过在标准变分下界中引入成对概率先验，我们的方法显著增强了对对手攻击的防御。我们的实验评估表明，这种方法被称为稳健的增广变分自动编码器(RAVEN)，在抵抗广泛认可的基准数据集上的敌意输入方面具有优越的性能。



## **43. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

DistriBlock：通过利用输出分布的特征来识别对抗性音频样本 cs.SD

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2305.17000v6) [paper-pdf](http://arxiv.org/pdf/2305.17000v6)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.

摘要: 敌意攻击可以误导自动语音识别(ASR)系统预测任意目标文本，从而构成明显的安全威胁。为了防止此类攻击，我们提出了DistriBlock，这是一种适用于任何ASR系统的有效检测策略，它预测每个时间步输出令牌上的概率分布。我们测量了该分布的一组特征：输出概率的中位数、最大值和最小值，分布的熵，以及关于后续时间步分布的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性数据和恶意数据观察到的特征，我们应用二进制分类器，包括简单的基于阈值的分类、这种分类器的集成和神经网络。通过对不同的ASR系统和语言数据集的广泛分析，我们证明了该方法的最高性能，在干净和有噪声的数据下，接收器操作特征曲线下的平均面积分别为99%和97%。为了评估我们方法的健壮性，我们证明了可以绕过DistriBlock的自适应攻击示例的噪声要大得多，这使得它们更容易通过过滤来检测，并为保持系统的健壮性创造了另一种途径。



## **44. Unveiling Privacy Vulnerabilities: Investigating the Role of Structure in Graph Data**

揭露隐私漏洞：调查结构在图形数据中的作用 cs.LG

In KDD'24; with full appendix

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18564v1) [paper-pdf](http://arxiv.org/pdf/2407.18564v1)

**Authors**: Hanyang Yuan, Jiarong Xu, Cong Wang, Ziqi Yang, Chunping Wang, Keting Yin, Yang Yang

**Abstract**: The public sharing of user information opens the door for adversaries to infer private data, leading to privacy breaches and facilitating malicious activities. While numerous studies have concentrated on privacy leakage via public user attributes, the threats associated with the exposure of user relationships, particularly through network structure, are often neglected. This study aims to fill this critical gap by advancing the understanding and protection against privacy risks emanating from network structure, moving beyond direct connections with neighbors to include the broader implications of indirect network structural patterns. To achieve this, we first investigate the problem of Graph Privacy Leakage via Structure (GPS), and introduce a novel measure, the Generalized Homophily Ratio, to quantify the various mechanisms contributing to privacy breach risks in GPS. Based on this insight, we develop a novel graph private attribute inference attack, which acts as a pivotal tool for evaluating the potential for privacy leakage through network structures under worst-case scenarios. To protect users' private data from such vulnerabilities, we propose a graph data publishing method incorporating a learnable graph sampling technique, effectively transforming the original graph into a privacy-preserving version. Extensive experiments demonstrate that our attack model poses a significant threat to user privacy, and our graph data publishing method successfully achieves the optimal privacy-utility trade-off compared to baselines.

摘要: 用户信息的公开共享为攻击者推断私人数据打开了大门，导致隐私被侵犯，并为恶意活动提供便利。虽然许多研究都集中在通过公共用户属性泄露隐私，但与暴露用户关系相关的威胁，特别是通过网络结构，往往被忽视。这项研究旨在通过促进对网络结构产生的隐私风险的理解和保护来填补这一关键空白，超越与邻居的直接连接，包括间接网络结构模式的更广泛影响。为此，我们首先研究了通过结构的图隐私泄露问题，并引入了一种新的度量方法--广义同伦比来量化GPS中导致隐私泄露风险的各种机制。基于这一观点，我们开发了一种新的图私有属性推理攻击，该攻击可以作为评估最坏情况下通过网络结构泄露隐私的可能性的关键工具。为了保护用户的私有数据免受此类漏洞的侵害，我们提出了一种结合了可学习的图采样技术的图数据发布方法，有效地将原始图转换为隐私保护版本。大量的实验表明，我们的攻击模型对用户隐私构成了严重的威胁，并且我们的图数据发布方法成功地实现了相对于基线的最优隐私效用权衡。



## **45. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

Janus界面：大型语言模型中的微调如何放大隐私风险 cs.CR

This work has been accepted by CCS 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2310.15469v3) [paper-pdf](http://arxiv.org/pdf/2310.15469v3)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang

**Abstract**: The rapid advancements of large language models (LLMs) have raised public concerns about the privacy leakage of personally identifiable information (PII) within their extensive training datasets. Recent studies have demonstrated that an adversary could extract highly sensitive privacy data from the training data of LLMs with carefully designed prompts. However, these attacks suffer from the model's tendency to hallucinate and catastrophic forgetting (CF) in the pre-training stage, rendering the veracity of divulged PIIs negligible. In our research, we propose a novel attack, Janus, which exploits the fine-tuning interface to recover forgotten PIIs from the pre-training data in LLMs. We formalize the privacy leakage problem in LLMs and explain why forgotten PIIs can be recovered through empirical analysis on open-source language models. Based upon these insights, we evaluate the performance of Janus on both open-source language models and two latest LLMs, i.e., GPT-3.5-Turbo and LLaMA-2-7b. Our experiment results show that Janus amplifies the privacy risks by over 10 times in comparison with the baseline and significantly outperforms the state-of-the-art privacy extraction attacks including prefix attacks and in-context learning (ICL). Furthermore, our analysis validates that existing fine-tuning APIs provided by OpenAI and Azure AI Studio are susceptible to our Janus attack, allowing an adversary to conduct such an attack at a low cost.

摘要: 大型语言模型(LLM)的快速发展引起了公众对其广泛训练数据集中个人身份信息(PII)隐私泄露的担忧。最近的研究表明，攻击者可以通过精心设计的提示从LLMS的训练数据中提取高度敏感的隐私数据。然而，这些攻击受到模型在预训练阶段的幻觉和灾难性遗忘(CF)的倾向的影响，使得泄露的PII的真实性可以忽略不计。在我们的研究中，我们提出了一种新的攻击，Janus，它利用微调接口从LLMS的训练前数据中恢复被遗忘的PII。我们形式化地描述了LLMS中的隐私泄露问题，并通过对开源语言模型的实证分析解释了为什么被遗忘的PII可以恢复。基于这些见解，我们评估了Janus在开源语言模型和两个最新的LLMS上的性能，即GPT-3.5-Turbo和Llama-2-7b。我们的实验结果表明，Janus将隐私风险放大了10倍以上，并且显著优于目前最先进的隐私提取攻击，包括前缀攻击和上下文中学习(ICL)。此外，我们的分析验证了OpenAI和Azure AI Studio提供的现有微调API容易受到我们的Janus攻击，允许对手以低成本进行此类攻击。



## **46. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

Dysca：评估LVLM感知能力的动态和可扩展基准 cs.CV

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2406.18849v2) [paper-pdf](http://arxiv.org/pdf/2406.18849v2)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 8 advanced open-source LVLMs with 10 checkpoints are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released in \url{https://github.com/Benchmark-Dysca/Dysca}.

摘要: 目前，人们已经提出了许多基准来评估大型视觉语言模型的感知能力。然而，大多数基准测试通过从现有数据集中选择图像来进行问题，从而导致潜在的数据泄漏。此外，这些基准只关注真实感风格的图像和干净的场景来评估LVLMS，而对多风格化的图像和噪声场景没有进行探索。为了应对这些挑战，我们提出了一种动态的、可扩展的基准测试DYSCA，用于利用合成图像来评估LVLMS。具体地说，我们利用稳定扩散并设计了一种基于规则的方法来动态生成新的图像、问题和相应的答案。我们考虑了51种图像风格，并在20个子任务中评估了感知能力。此外，我们还在4个场景(即廉洁、腐败、打印攻击和对抗性攻击)和3个问题类型(即多项选择、对错和自由形式)下进行了评估。多亏了生成性范式，Dysca成为了一个可伸缩的基准，可以轻松添加新的子任务和场景。在Dysca上对8个具有10个检查点的高级开源LVLMS进行了评估，揭示了当前LVLMS的缺陷。该基准测试在\url{https://github.com/Benchmark-Dysca/Dysca}.中发布



## **47. Machine Unlearning using a Multi-GAN based Model**

使用基于多GAN的模型的机器去学习 cs.LG

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18467v1) [paper-pdf](http://arxiv.org/pdf/2407.18467v1)

**Authors**: Amartya Hatua, Trung T. Nguyen, Andrew H. Sung

**Abstract**: This article presents a new machine unlearning approach that utilizes multiple Generative Adversarial Network (GAN) based models. The proposed method comprises two phases: i) data reorganization in which synthetic data using the GAN model is introduced with inverted class labels of the forget datasets, and ii) fine-tuning the pre-trained model. The GAN models consist of two pairs of generators and discriminators. The generator discriminator pairs generate synthetic data for the retain and forget datasets. Then, a pre-trained model is utilized to get the class labels of the synthetic datasets. The class labels of synthetic and original forget datasets are inverted. Finally, all combined datasets are used to fine-tune the pre-trained model to get the unlearned model. We have performed the experiments on the CIFAR-10 dataset and tested the unlearned models using Membership Inference Attacks (MIA). The inverted class labels procedure and synthetically generated data help to acquire valuable information that enables the model to outperform state-of-the-art models and other standard unlearning classifiers.

摘要: 提出了一种基于多生成性对抗网络(GAN)模型的机器遗忘方法。该方法包括两个阶段：i)数据重组，其中使用GAN模型的合成数据被引入带有倒排的忘记数据集的类别标签；ii)微调预先训练的模型。GaN模型由两对生成器和鉴别器组成。生成器鉴别器对为保留和忘记数据集生成合成数据。然后，利用预先训练好的模型得到合成数据集的类标签。对合成的和原始的忘记数据集的分类标签进行倒置。最后，使用所有组合的数据集来微调预先训练的模型，以获得未学习的模型。我们在CIFAR-10数据集上进行了实验，并使用隶属度推理攻击(MIA)对未学习模型进行了测试。倒置的分类标签程序和合成生成的数据有助于获取有价值的信息，使该模型的表现优于最先进的模型和其他标准的遗忘分类器。



## **48. Disrupting Diffusion: Token-Level Attention Erasure Attack against Diffusion-based Customization**

扰乱扩散：针对基于扩散的定制的代币级注意力擦除攻击 cs.CV

Accepted by ACM MM2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2405.20584v2) [paper-pdf](http://arxiv.org/pdf/2405.20584v2)

**Authors**: Yisu Liu, Jinyang An, Wanqian Zhang, Dayan Wu, Jingzi Gu, Zheng Lin, Weiping Wang

**Abstract**: With the development of diffusion-based customization methods like DreamBooth, individuals now have access to train the models that can generate their personalized images. Despite the convenience, malicious users have misused these techniques to create fake images, thereby triggering a privacy security crisis. In light of this, proactive adversarial attacks are proposed to protect users against customization. The adversarial examples are trained to distort the customization model's outputs and thus block the misuse. In this paper, we propose DisDiff (Disrupting Diffusion), a novel adversarial attack method to disrupt the diffusion model outputs. We first delve into the intrinsic image-text relationships, well-known as cross-attention, and empirically find that the subject-identifier token plays an important role in guiding image generation. Thus, we propose the Cross-Attention Erasure module to explicitly "erase" the indicated attention maps and disrupt the text guidance. Besides,we analyze the influence of the sampling process of the diffusion model on Projected Gradient Descent (PGD) attack and introduce a novel Merit Sampling Scheduler to adaptively modulate the perturbation updating amplitude in a step-aware manner. Our DisDiff outperforms the state-of-the-art methods by 12.75% of FDFR scores and 7.25% of ISM scores across two facial benchmarks and two commonly used prompts on average.

摘要: 随着像DreamBooth这样基于扩散的定制方法的发展，个人现在可以训练能够生成他们个性化图像的模型。尽管很方便，但恶意用户滥用这些技术创造了虚假图像，从而引发了隐私安全危机。有鉴于此，提出了主动对抗性攻击来保护用户免受定制。对抗性的例子被训练来扭曲定制模型的输出，从而阻止误用。在本文中，我们提出了一种新的对抗性攻击方法DisDiff(破坏扩散)来破坏扩散模型的输出。我们首先深入研究了内在的图文关系，即众所周知的交叉注意，并经验地发现，主体-标识符号在引导图像生成方面发挥着重要作用。因此，我们提出了交叉注意擦除模块来显式地“擦除”所指示的注意地图并扰乱文本引导。此外，我们还分析了扩散模型的采样过程对投影梯度下降(PGD)攻击的影响，并引入了一种新的优点采样调度器来以步长感知的方式自适应地调制扰动更新幅度。在两个面部基准和两个常用提示上，我们的DisDiff平均比最先进的方法高出12.75%的FDFR分数和7.25%的ISM分数。



## **49. Regret-Optimal Defense Against Stealthy Adversaries: A System Level Approach**

针对潜行对手的遗憾最佳防御：系统级方法 eess.SY

Accepted, IEEE Conference on Decision and Control (CDC), 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18448v1) [paper-pdf](http://arxiv.org/pdf/2407.18448v1)

**Authors**: Hiroyasu Tsukamoto, Joudi Hajar, Soon-Jo Chung, Fred Y. Hadaegh

**Abstract**: Modern control designs in robotics, aerospace, and cyber-physical systems heavily depend on real-world data obtained through the system outputs. In the face of system faults and malicious attacks, however, these outputs can be compromised to misrepresent some portion of the system information that critically affects their secure and trustworthy operation. In this paper, we introduce a novel regret-optimal control framework for designing controllers that render a linear system robust against stealthy attacks, including sensor and actuator attacks, as well as external disturbances. In particular, we establish (a) a convex optimization-based system metric to quantify the regret with the worst-case stealthy attack (the true performance minus the optimal performance in hindsight with the knowledge of the stealthy attack), which improves and adaptively interpolates $\mathcal{H}_2$ and $\mathcal{H}_{\infty}$ norms in the presence of stealthy adversaries, (b) an optimization problem for minimizing the regret of 1 expressed in the system level parameterization, which is useful for its localized and distributed implementation in large-scale systems, and (c) a rank-constrained optimization problem (i.e., optimization with a convex objective subject to convex constraints and rank constraints) equivalent to the optimization problem of (b). Finally, we conduct a numerical simulation which showcases the effectiveness of our approach.

摘要: 机器人、航空航天和计算机物理系统中的现代控制设计在很大程度上依赖于通过系统输出获得的真实世界数据。然而，面对系统故障和恶意攻击，这些输出可能会受到损害，从而歪曲系统信息的某些部分，从而严重影响其安全和可信的操作。在本文中，我们介绍了一种新的后悔最优控制框架，用于设计控制器，使线性系统对隐身攻击，包括传感器和执行器攻击，以及外部干扰具有鲁棒性。具体地，我们建立了(A)基于凸优化的系统度量来量化最坏情况下的隐蔽攻击的遗憾(真实性能减去具有隐形攻击知识的后见之明的最优性能)，它改进并自适应地内插在隐身对手存在的情况下的$\数学{H}_2$和$\数学{H}_(Inty)$范数；(B)以系统级参数化表示的最小化1的遗憾的优化问题，这对于其在大规模系统中的局部和分布式实现是有用的；以及(C)秩约束优化问题(即，具有凸约束和秩约束的凸目标最优化)等价于(B)的最优化问题。最后，我们进行了数值模拟，验证了该方法的有效性。



## **50. Human-Interpretable Adversarial Prompt Attack on Large Language Models with Situational Context**

具有情境上下文的大型语言模型的人类可解释对抗提示攻击 cs.CL

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.14644v2) [paper-pdf](http://arxiv.org/pdf/2407.14644v2)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous research on testing the vulnerabilities in Large Language Models (LLMs) using adversarial attacks has primarily focused on nonsensical prompt injections, which are easily detected upon manual or automated review (e.g., via byte entropy). However, the exploration of innocuous human-understandable malicious prompts augmented with adversarial injections remains limited. In this research, we explore converting a nonsensical suffix attack into a sensible prompt via a situation-driven contextual re-writing. This allows us to show suffix conversion without any gradients, using only LLMs to perform the attacks, and thus better understand the scope of possible risks. We combine an independent, meaningful adversarial insertion and situations derived from movies to check if this can trick an LLM. The situations are extracted from the IMDB dataset, and prompts are defined following a few-shot chain-of-thought prompting. Our approach demonstrates that a successful situation-driven attack can be executed on both open-source and proprietary LLMs. We find that across many LLMs, as few as 1 attempt produces an attack and that these attacks transfer between LLMs.

摘要: 之前关于使用对抗性攻击测试大型语言模型(LLM)中的漏洞的研究主要集中在无意义的提示注入上，这些注入很容易通过手动或自动审查(例如，通过字节熵)检测到。然而，通过恶意注入增强无害的人类可理解的恶意提示的探索仍然有限。在这项研究中，我们探索通过情景驱动的语境重写将无意义的后缀攻击转化为合理的提示。这使我们能够显示没有任何梯度的后缀转换，仅使用LLM来执行攻击，从而更好地了解可能风险的范围。我们结合了一个独立的、有意义的敌意插入和来自电影的情况来检查这是否可以欺骗LLM。情况是从IMDB数据集中提取的，提示是在几个镜头的思维链提示之后定义的。我们的方法表明，成功的情境驱动攻击可以在开源和专有LLM上执行。我们发现，在许多LLM中，只有1次尝试就会产生攻击，并且这些攻击会在LLM之间传输。



