# Latest Adversarial Attack Papers
**update at 2025-02-05 12:10:49**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. OVERTHINKING: Slowdown Attacks on Reasoning LLMs**

过度思考：对推理LLM的缓慢攻击 cs.LG

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02542v1) [paper-pdf](http://arxiv.org/pdf/2502.02542v1)

**Authors**: Abhinav Kumar, Jaechul Roh, Ali Naseh, Marzena Karpinska, Mohit Iyyer, Amir Houmansadr, Eugene Bagdasarian

**Abstract**: We increase overhead for applications that rely on reasoning LLMs-we force models to spend an amplified number of reasoning tokens, i.e., "overthink", to respond to the user query while providing contextually correct answers. The adversary performs an OVERTHINK attack by injecting decoy reasoning problems into the public content that is used by the reasoning LLM (e.g., for RAG applications) during inference time. Due to the nature of our decoy problems (e.g., a Markov Decision Process), modified texts do not violate safety guardrails. We evaluated our attack across closed-(OpenAI o1, o1-mini, o3-mini) and open-(DeepSeek R1) weights reasoning models on the FreshQA and SQuAD datasets. Our results show up to 46x slowdown and high transferability of the attack across models. To protect applications, we discuss and implement defenses leveraging LLM-based and system design approaches. Finally, we discuss societal, financial, and energy impacts of OVERTHINK attack which could amplify the costs for third party applications operating reasoning models.

摘要: 我们增加了依赖推理LLM的应用程序的开销-我们迫使模型花费更多的推理标记，即“过度思考”，以响应用户查询，同时提供上下文正确的答案。敌手通过在推理时间期间将诱骗推理问题注入到推理LLM(例如，用于RAG应用)使用的公共内容中来执行过度思考攻击。由于我们的诱饵问题的性质(例如，马尔可夫决策过程)，修改后的文本不会违反安全护栏。我们在FreshQA和LONG数据集上评估了我们的攻击，跨越了封闭(OpenAI o1，o1-mini，o3-mini)和开放(DeepSeek R1)权重推理模型。我们的结果显示，攻击的速度最高可达46倍，并且跨模型的可转移性很高。为了保护应用程序，我们讨论并实施了利用基于LLM和系统设计方法的防御措施。最后，我们讨论了过度思考攻击的社会、金融和能源影响，这种攻击可能会放大第三方应用程序运行推理模型的成本。



## **2. Uncertainty Quantification for Collaborative Object Detection Under Adversarial Attacks**

对抗攻击下协作对象检测的不确定性量化 cs.CV

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02537v1) [paper-pdf](http://arxiv.org/pdf/2502.02537v1)

**Authors**: Huiqun Huang, Cong Chen, Jean-Philippe Monteuuis, Jonathan Petit, Fei Miao

**Abstract**: Collaborative Object Detection (COD) and collaborative perception can integrate data or features from various entities, and improve object detection accuracy compared with individual perception. However, adversarial attacks pose a potential threat to the deep learning COD models, and introduce high output uncertainty. With unknown attack models, it becomes even more challenging to improve COD resiliency and quantify the output uncertainty for highly dynamic perception scenes such as autonomous vehicles. In this study, we propose the Trusted Uncertainty Quantification in Collaborative Perception framework (TUQCP). TUQCP leverages both adversarial training and uncertainty quantification techniques to enhance the adversarial robustness of existing COD models. More specifically, TUQCP first adds perturbations to the shared information of randomly selected agents during object detection collaboration by adversarial training. TUQCP then alleviates the impacts of adversarial attacks by providing output uncertainty estimation through learning-based module and uncertainty calibration through conformal prediction. Our framework works for early and intermediate collaboration COD models and single-agent object detection models. We evaluate TUQCP on V2X-Sim, a comprehensive collaborative perception dataset for autonomous driving, and demonstrate a 80.41% improvement in object detection accuracy compared to the baselines under the same adversarial attacks. TUQCP demonstrates the importance of uncertainty quantification to COD under adversarial attacks.

摘要: [TencentCloudSDKException] code:InternalError.BackendTimeout message:Backend timeout, please retry it later requestId:bef2bd0d-84a4-4fcf-865c-275c66ed979d



## **3. The TIP of the Iceberg: Revealing a Hidden Class of Task-in-Prompt Adversarial Attacks on LLMs**

冰山的提示：揭示对LLM的一类隐藏的即时任务对抗性攻击 cs.CR

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2501.18626v3) [paper-pdf](http://arxiv.org/pdf/2501.18626v3)

**Authors**: Sergey Berezin, Reza Farahbakhsh, Noel Crespi

**Abstract**: We present a novel class of jailbreak adversarial attacks on LLMs, termed Task-in-Prompt (TIP) attacks. Our approach embeds sequence-to-sequence tasks (e.g., cipher decoding, riddles, code execution) into the model's prompt to indirectly generate prohibited inputs. To systematically assess the effectiveness of these attacks, we introduce the PHRYGE benchmark. We demonstrate that our techniques successfully circumvent safeguards in six state-of-the-art language models, including GPT-4o and LLaMA 3.2. Our findings highlight critical weaknesses in current LLM safety alignments and underscore the urgent need for more sophisticated defence strategies.   Warning: this paper contains examples of unethical inquiries used solely for research purposes.

摘要: 我们提出了一类新型的针对LLM的越狱对抗攻击，称为提示任务（TIP）攻击。我们的方法嵌入序列到序列任务（例如，密码解码、谜语、代码执行）到模型的提示中，以间接生成禁止的输入。为了系统性评估这些攻击的有效性，我们引入了PHRYGE基准。我们证明我们的技术成功规避了六种最先进语言模型（包括GPT-4 o和LLaMA 3.2）中的保护措施。我们的研究结果凸显了当前LLM安全调整中的关键弱点，并强调了对更复杂防御策略的迫切需要。   警告：本文包含仅用于研究目的的不道德调查的例子。



## **4. Medical Multimodal Model Stealing Attacks via Adversarial Domain Alignment**

医学多模式模型通过对抗领域对齐窃取攻击 cs.CR

Accepted at AAAI 2025

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02438v1) [paper-pdf](http://arxiv.org/pdf/2502.02438v1)

**Authors**: Yaling Shen, Zhixiong Zhuang, Kun Yuan, Maria-Irina Nicolae, Nassir Navab, Nicolas Padoy, Mario Fritz

**Abstract**: Medical multimodal large language models (MLLMs) are becoming an instrumental part of healthcare systems, assisting medical personnel with decision making and results analysis. Models for radiology report generation are able to interpret medical imagery, thus reducing the workload of radiologists. As medical data is scarce and protected by privacy regulations, medical MLLMs represent valuable intellectual property. However, these assets are potentially vulnerable to model stealing, where attackers aim to replicate their functionality via black-box access. So far, model stealing for the medical domain has focused on classification; however, existing attacks are not effective against MLLMs. In this paper, we introduce Adversarial Domain Alignment (ADA-STEAL), the first stealing attack against medical MLLMs. ADA-STEAL relies on natural images, which are public and widely available, as opposed to their medical counterparts. We show that data augmentation with adversarial noise is sufficient to overcome the data distribution gap between natural images and the domain-specific distribution of the victim MLLM. Experiments on the IU X-RAY and MIMIC-CXR radiology datasets demonstrate that Adversarial Domain Alignment enables attackers to steal the medical MLLM without any access to medical data.

摘要: 医疗多模式大型语言模型(MLLMS)正在成为医疗保健系统的重要组成部分，帮助医务人员进行决策和结果分析。放射学报告生成模型能够解释医学图像，从而减少了放射科医生的工作量。由于医疗数据稀缺，而且受到隐私法规的保护，医疗MLLM代表着宝贵的知识产权。然而，这些资产可能容易受到模型窃取的攻击，攻击者的目标是通过黑盒访问来复制它们的功能。到目前为止，针对医学领域的模型窃取主要集中在分类上，然而，现有的攻击对MLLMS并不有效。在本文中，我们介绍了第一个针对医学MLLM的窃取攻击--对抗性领域对齐(ADA-Steal)。Ada-steal依赖于自然图像，这些图像是公开的，可以广泛使用，而不是医学上的同行。我们证明了使用对抗性噪声的数据增强足以克服自然图像和受害者MLLM的特定领域分布之间的数据分布差距。在Iu-X-Ray和MIMIC-CXR放射学数据集上的实验表明，对抗性领域对齐使攻击者能够在不访问任何医疗数据的情况下窃取医疗MLLM。



## **5. Rule-ATT&CK Mapper (RAM): Mapping SIEM Rules to TTPs Using LLMs**

规则-ATA & CK映射器（RAM）：使用LLM将SIEM规则映射到TTP cs.CR

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02337v1) [paper-pdf](http://arxiv.org/pdf/2502.02337v1)

**Authors**: Prasanna N. Wudali, Moshe Kravchik, Ehud Malul, Parth A. Gandhi, Yuval Elovici, Asaf Shabtai

**Abstract**: The growing frequency of cyberattacks has heightened the demand for accurate and efficient threat detection systems. SIEM platforms are important for analyzing log data and detecting adversarial activities through rule-based queries, also known as SIEM rules. The efficiency of the threat analysis process relies heavily on mapping these SIEM rules to the relevant attack techniques in the MITRE ATT&CK framework. Inaccurate annotation of SIEM rules can result in the misinterpretation of attacks, increasing the likelihood that threats will be overlooked. Existing solutions for annotating SIEM rules with MITRE ATT&CK technique labels have notable limitations: manual annotation of SIEM rules is both time-consuming and prone to errors, and ML-based approaches mainly focus on annotating unstructured free text sources rather than structured data like SIEM rules. Structured data often contains limited information, further complicating the annotation process and making it a challenging task. To address these challenges, we propose Rule-ATT&CK Mapper (RAM), a novel framework that leverages LLMs to automate the mapping of structured SIEM rules to MITRE ATT&CK techniques. RAM's multi-stage pipeline, which was inspired by the prompt chaining technique, enhances mapping accuracy without requiring LLM pre-training or fine-tuning. Using the Splunk Security Content dataset, we evaluate RAM's performance using several LLMs, including GPT-4-Turbo, Qwen, IBM Granite, and Mistral. Our evaluation highlights GPT-4-Turbo's superior performance, which derives from its enriched knowledge base, and an ablation study emphasizes the importance of external contextual knowledge in overcoming the limitations of LLMs' implicit knowledge for domain-specific tasks. These findings demonstrate RAM's potential in automating cybersecurity workflows and provide valuable insights for future advancements in this field.

摘要: 网络攻击的频率越来越高，这提高了对准确高效的威胁检测系统的需求。SIEM平台对于通过基于规则的查询(也称为SIEM规则)分析日志数据和检测敌对活动非常重要。威胁分析过程的效率在很大程度上依赖于将这些SIEM规则映射到MITRE ATT&CK框架中的相关攻击技术。对SIEM规则的不准确注释可能会导致对攻击的误解，从而增加威胁被忽视的可能性。现有的使用MITRE ATT&CK技术标签标注SIEM规则的解决方案有明显的局限性：手工标注SIEM规则既耗时又容易出错，基于ML的方法主要专注于标注非结构化自由文本源而不是像SIEM规则这样的结构化数据。结构化数据通常包含有限的信息，这使注释过程进一步复杂化，并使其成为一项具有挑战性的任务。为了应对这些挑战，我们提出了规则-ATT&CK映射器(RAM)，这是一个新的框架，它利用LLMS来自动将结构化SIEM规则映射到MITRE ATT&CK技术。RAM的多级流水线的灵感来自于快速链接技术，无需LLM预训练或微调即可提高映射精度。使用Splunk Security内容数据集，我们使用几个LLM来评估RAM的性能，包括GPT-4-Turbo、Qwen、IBM Granite和Mistral。我们的评估突出了GPT-4-Turbo的卓越性能，这源于其丰富的知识库，而一项消融研究强调了外部上下文知识在克服LLMS的隐含知识对特定领域任务的限制方面的重要性。这些发现展示了RAM在自动化网络安全工作流程方面的潜力，并为该领域的未来发展提供了有价值的见解。



## **6. FRAUD-RLA: A new reinforcement learning adversarial attack against credit card fraud detection**

FARUD-RLA：针对信用卡欺诈检测的新强化学习对抗攻击 cs.LG

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02290v1) [paper-pdf](http://arxiv.org/pdf/2502.02290v1)

**Authors**: Daniele Lunghi, Yannick Molinghen, Alkis Simitsis, Tom Lenaerts, Gianluca Bontempi

**Abstract**: Adversarial attacks pose a significant threat to data-driven systems, and researchers have spent considerable resources studying them. Despite its economic relevance, this trend largely overlooked the issue of credit card fraud detection. To address this gap, we propose a new threat model that demonstrates the limitations of existing attacks and highlights the necessity to investigate new approaches. We then design a new adversarial attack for credit card fraud detection, employing reinforcement learning to bypass classifiers. This attack, called FRAUD-RLA, is designed to maximize the attacker's reward by optimizing the exploration-exploitation tradeoff and working with significantly less required knowledge than competitors. Our experiments, conducted on three different heterogeneous datasets and against two fraud detection systems, indicate that FRAUD-RLA is effective, even considering the severe limitations imposed by our threat model.

摘要: 对抗性攻击对数据驱动系统构成重大威胁，研究人员花费了大量资源来研究它们。尽管具有经济相关性，但这一趋势在很大程度上忽视了信用卡欺诈检测问题。为了弥补这一差距，我们提出了一种新的威胁模型，该模型展示了现有攻击的局限性，并强调了研究新方法的必要性。然后，我们设计了一种用于信用卡欺诈检测的新对抗攻击，采用强化学习来绕过分类器。这种名为FRAUP-RLA的攻击旨在通过优化探索与利用的权衡以及所需的知识比竞争对手少得多来最大化攻击者的回报。我们在三个不同的异类数据集上并针对两个欺诈检测系统进行的实验表明，即使考虑到我们的威胁模型所施加的严重限制，FARUD-RLA仍然有效。



## **7. Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization**

双流：通过野外级联流优化进行可转移的多目标、实例不可知攻击 cs.CV

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02096v1) [paper-pdf](http://arxiv.org/pdf/2502.02096v1)

**Authors**: Yixiao Chen, Shikun Sun, Jianshu Li, Ruoyu Li, Zhe Li, Junliang Xing

**Abstract**: Adversarial attacks are widely used to evaluate model robustness, and in black-box scenarios, the transferability of these attacks becomes crucial. Existing generator-based attacks have excellent generalization and transferability due to their instance-agnostic nature. However, when training generators for multi-target tasks, the success rate of transfer attacks is relatively low due to the limitations of the model's capacity. To address these challenges, we propose a novel Dual-Flow framework for multi-target instance-agnostic adversarial attacks, utilizing Cascading Distribution Shift Training to develop an adversarial velocity function. Extensive experiments demonstrate that Dual-Flow significantly improves transferability over previous multi-target generative attacks. For example, it increases the success rate from Inception-v3 to ResNet-152 by 34.58%. Furthermore, our attack method, such as adversarially trained models, shows substantially stronger robustness against defense mechanisms.

摘要: 对抗性攻击被广泛用于评估模型的稳健性，在黑盒场景中，这些攻击的可转移性变得至关重要。现有的基于生成器的攻击由于其与实例无关的性质而具有良好的泛化和可转移性。然而，当训练多目标任务的生成器时，由于模型能力的限制，转移攻击的成功率相对较低。为了应对这些挑战，我们提出了一种新的针对多目标实例不可知对手攻击的双流框架，利用级联分布平移训练来开发对手攻击的速度函数。大量实验表明，与以往的多目标生成性攻击相比，双流攻击显著提高了可转移性。例如，它将从初始版本v3到ResNet-152的成功率提高了34.58%。此外，我们的攻击方法，如对抗性训练模型，对防御机制显示出更强的稳健性。



## **8. Model Supply Chain Poisoning: Backdooring Pre-trained Models via Embedding Indistinguishability**

模型供应链中毒：通过嵌入不可分割性对预训练模型进行后门 cs.CR

ACM Web Conference 2025 (Oral)

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2401.15883v3) [paper-pdf](http://arxiv.org/pdf/2401.15883v3)

**Authors**: Hao Wang, Shangwei Guo, Jialing He, Hangcheng Liu, Tianwei Zhang, Tao Xiang

**Abstract**: Pre-trained models (PTMs) are widely adopted across various downstream tasks in the machine learning supply chain. Adopting untrustworthy PTMs introduces significant security risks, where adversaries can poison the model supply chain by embedding hidden malicious behaviors (backdoors) into PTMs. However, existing backdoor attacks to PTMs can only achieve partially task-agnostic and the embedded backdoors are easily erased during the fine-tuning process. This makes it challenging for the backdoors to persist and propagate through the supply chain. In this paper, we propose a novel and severer backdoor attack, TransTroj, which enables the backdoors embedded in PTMs to efficiently transfer in the model supply chain. In particular, we first formalize this attack as an indistinguishability problem between poisoned and clean samples in the embedding space. We decompose embedding indistinguishability into pre- and post-indistinguishability, representing the similarity of the poisoned and reference embeddings before and after the attack. Then, we propose a two-stage optimization that separately optimizes triggers and victim PTMs to achieve embedding indistinguishability. We evaluate TransTroj on four PTMs and six downstream tasks. Experimental results show that our method significantly outperforms SOTA task-agnostic backdoor attacks -- achieving nearly 100% attack success rate on most downstream tasks -- and demonstrates robustness under various system settings. Our findings underscore the urgent need to secure the model supply chain against such transferable backdoor attacks. The code is available at https://github.com/haowang-cqu/TransTroj .

摘要: 预训练模型(PTM)广泛应用于机器学习供应链中的各种下游任务。采用不可信的PTMS会带来严重的安全风险，攻击者可以通过在PTMS中嵌入隐藏的恶意行为(后门)来毒化模型供应链。然而，现有的对PTMS的后门攻击只能实现部分任务无关，并且嵌入的后门在微调过程中很容易被擦除。这使得后门在供应链中的持续和传播变得具有挑战性。在本文中，我们提出了一种新的更严重的后门攻击，TransTroj，它使得嵌入PTMS的后门能够在模型供应链中有效地转移。特别地，我们首先将这种攻击形式化为嵌入空间中有毒样本和干净样本之间的不可区分问题。我们将嵌入不可区分性分解为攻击前后的不可区分性，表示攻击前后中毒嵌入和参考嵌入的相似性。然后，我们提出了一种两阶段优化方法，分别对触发者和受害者PTM进行优化，以达到嵌入不可区分的目的。我们在四个PTM和六个下游任务上对TransTroj进行了评估。实验结果表明，我们的方法显著优于SOTA任务无关的后门攻击--在大多数下游任务上获得近100%的攻击成功率--并在各种系统设置下表现出健壮性。我们的发现突显出，迫切需要确保模型供应链免受这种可转移的后门攻击。代码可在https://github.com/haowang-cqu/TransTroj上获得。



## **9. Multi-Domain Graph Foundation Models: Robust Knowledge Transfer via Topology Alignment**

多领域图基础模型：通过布局对齐实现稳健的知识转移 cs.SI

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02017v1) [paper-pdf](http://arxiv.org/pdf/2502.02017v1)

**Authors**: Shuo Wang, Bokui Wang, Zhixiang Shen, Boyan Deng, Zhao Kang

**Abstract**: Recent advances in CV and NLP have inspired researchers to develop general-purpose graph foundation models through pre-training across diverse domains. However, a fundamental challenge arises from the substantial differences in graph topologies across domains. Additionally, real-world graphs are often sparse and prone to noisy connections and adversarial attacks. To address these issues, we propose the Multi-Domain Graph Foundation Model (MDGFM), a unified framework that aligns and leverages cross-domain topological information to facilitate robust knowledge transfer. MDGFM bridges different domains by adaptively balancing features and topology while refining original graphs to eliminate noise and align topological structures. To further enhance knowledge transfer, we introduce an efficient prompt-tuning approach. By aligning topologies, MDGFM not only improves multi-domain pre-training but also enables robust knowledge transfer to unseen domains. Theoretical analyses provide guarantees of MDGFM's effectiveness and domain generalization capabilities. Extensive experiments on both homophilic and heterophilic graph datasets validate the robustness and efficacy of our method.

摘要: CV和NLP的最新进展启发了研究人员通过跨不同领域的预训练来开发通用的图形基础模型。然而，一个根本的挑战来自于跨域的图形拓扑的巨大差异。此外，真实世界的图形通常是稀疏的，容易受到噪声连接和敌意攻击。为了解决这些问题，我们提出了多领域图基础模型(MDGFM)，这是一个统一的框架，对齐和利用跨域拓扑信息来促进健壮的知识转移。MDGFM通过自适应地平衡特征和拓扑来桥接不同的域，同时优化原始图以消除噪声并对齐拓扑结构。为了进一步加强知识转移，我们引入了一种有效的即时调整方法。通过对齐拓扑，MDGFM不仅改善了多领域的预训练，还使知识能够稳健地转移到看不见的领域。理论分析为MDGFM的有效性和领域泛化能力提供了保证。在同嗜性和异嗜性图形数据集上的大量实验验证了该方法的稳健性和有效性。



## **10. Evaluating the Robustness of the "Ensemble Everything Everywhere" Defense**

评估“无处不在”防御的稳健性 cs.LG

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2411.14834v2) [paper-pdf](http://arxiv.org/pdf/2411.14834v2)

**Authors**: Jie Zhang, Christian Schlarmann, Kristina Nikolić, Nicholas Carlini, Francesco Croce, Matthias Hein, Florian Tramèr

**Abstract**: Ensemble everything everywhere is a defense to adversarial examples that was recently proposed to make image classifiers robust. This defense works by ensembling a model's intermediate representations at multiple noisy image resolutions, producing a single robust classification. This defense was shown to be effective against multiple state-of-the-art attacks. Perhaps even more convincingly, it was shown that the model's gradients are perceptually aligned: attacks against the model produce noise that perceptually resembles the targeted class.   In this short note, we show that this defense is not robust to adversarial attack. We first show that the defense's randomness and ensembling method cause severe gradient masking. We then use standard adaptive attack techniques to reduce the defense's robust accuracy from 48% to 14% on CIFAR-100 and from 62% to 11% on CIFAR-10, under the $\ell_\infty$-norm threat model with $\varepsilon=8/255$.

摘要: 包容无处不在的一切是对最近提出的对抗性示例的防御，以使图像分类器稳健。这种防御的工作原理是以多个有噪图像分辨率集成模型的中间表示，产生单个稳健的分类。事实证明，这种防御措施对多种最先进的攻击有效。也许更令人信服的是，它表明模型的梯度在感知上是对齐的：对模型的攻击会产生在感知上类似于目标类的噪音。   在这篇简短的注释中，我们表明这种防御对对抗性攻击并不强大。我们首先表明防御的随机性和集成方法会导致严重的梯度掩蔽。然后，在$\ell_\infty$-norm威胁模型下，我们使用标准的自适应攻击技术将CIFAR-100上的防御鲁棒准确性从48%降低到14%，CIFAR-10上的防御鲁棒准确性从62%降低到11%，$\varepð =8/255$。



## **11. Adversarial Reasoning at Jailbreaking Time**

越狱时的对抗推理 cs.LG

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01633v1) [paper-pdf](http://arxiv.org/pdf/2502.01633v1)

**Authors**: Mahdi Sabbaghi, Paul Kassianik, George Pappas, Yaron Singer, Amin Karbasi, Hamed Hassani

**Abstract**: As large language models (LLMs) are becoming more capable and widespread, the study of their failure cases is becoming increasingly important. Recent advances in standardizing, measuring, and scaling test-time compute suggest new methodologies for optimizing models to achieve high performance on hard tasks. In this paper, we apply these advances to the task of model jailbreaking: eliciting harmful responses from aligned LLMs. We develop an adversarial reasoning approach to automatic jailbreaking via test-time computation that achieves SOTA attack success rates (ASR) against many aligned LLMs, even the ones that aim to trade inference-time compute for adversarial robustness. Our approach introduces a new paradigm in understanding LLM vulnerabilities, laying the foundation for the development of more robust and trustworthy AI systems.

摘要: 随着大型语言模型（LLM）变得越来越强大和广泛，对其失败案例的研究变得越来越重要。标准化、测量和扩展测试时计算方面的最新进展为优化模型以在硬任务中实现高性能提出了新的方法。在本文中，我们将这些进展应用于模型越狱的任务：从对齐的LLM中引发有害反应。我们开发了一种通过测试时计算自动越狱的对抗推理方法，该方法针对许多对齐的LLM，即使是那些旨在以推理时计算为对抗鲁棒性的LLM，也可以实现SOTA攻击成功率（ASB）。我们的方法引入了理解LLM漏洞的新范式，为开发更强大、更值得信赖的人工智能系统奠定了基础。



## **12. Robust-LLaVA: On the Effectiveness of Large-Scale Robust Image Encoders for Multi-modal Large Language Models**

Robust-LLaVA：关于大规模鲁棒图像编码器对多模式大型语言模型的有效性 cs.CV

Under Review

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01576v1) [paper-pdf](http://arxiv.org/pdf/2502.01576v1)

**Authors**: Hashmat Shadab Malik, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar, Fahad Khan, Salman Khan

**Abstract**: Multi-modal Large Language Models (MLLMs) excel in vision-language tasks but remain vulnerable to visual adversarial perturbations that can induce hallucinations, manipulate responses, or bypass safety mechanisms. Existing methods seek to mitigate these risks by applying constrained adversarial fine-tuning to CLIP vision encoders on ImageNet-scale data, ensuring their generalization ability is preserved. However, this limited adversarial training restricts robustness and broader generalization. In this work, we explore an alternative approach of leveraging existing vision classification models that have been adversarially pre-trained on large-scale data. Our analysis reveals two principal contributions: (1) the extensive scale and diversity of adversarial pre-training enables these models to demonstrate superior robustness against diverse adversarial threats, ranging from imperceptible perturbations to advanced jailbreaking attempts, without requiring additional adversarial training, and (2) end-to-end MLLM integration with these robust models facilitates enhanced adaptation of language components to robust visual features, outperforming existing plug-and-play methodologies on complex reasoning tasks. Through systematic evaluation across visual question-answering, image captioning, and jail-break attacks, we demonstrate that MLLMs trained with these robust models achieve superior adversarial robustness while maintaining favorable clean performance. Our framework achieves 2x and 1.5x average robustness gains in captioning and VQA tasks, respectively, and delivers over 10% improvement against jailbreak attacks. Code and pretrained models will be available at https://github.com/HashmatShadab/Robust-LLaVA.

摘要: 多模式大语言模型(MLLMS)在视觉-语言任务中表现出色，但仍然容易受到视觉对抗性扰动的影响，这些扰动可能会导致幻觉、操纵反应或绕过安全机制。现有的方法试图通过对ImageNet尺度数据上的裁剪视觉编码器应用受限的对抗性微调来缓解这些风险，以确保它们的泛化能力得到保护。然而，这种有限的对抗性训练限制了健壮性和更广泛的泛化。在这项工作中，我们探索了一种替代方法，利用现有的视觉分类模型，这些模型已经在大规模数据上进行了相反的预训练。我们的分析揭示了两个主要贡献：(1)对抗性预训练的广泛规模和多样性使这些模型能够在不需要额外的对抗性训练的情况下，对从不可察觉的扰动到高级越狱尝试等不同的对抗性威胁表现出优越的健壮性；(2)端到端MLLM与这些健壮的模型的集成促进了语言成分对健壮视觉特征的增强适应，在复杂推理任务中的表现优于现有的即插即用方法。通过对视觉问答、图像字幕和越狱攻击的系统评估，我们证明了使用这些健壮模型训练的MLLMS在保持良好的干净性能的同时，获得了优越的对手健壮性。我们的框架在字幕和VQA任务中分别获得了2倍和1.5倍的平均健壮性提升，并在抵御越狱攻击方面提供了超过10%的改进。代码和预先培训的模型将在https://github.com/HashmatShadab/Robust-LLaVA.上提供



## **13. Quantum Quandaries: Unraveling Encoding Vulnerabilities in Quantum Neural Networks**

量子困境：解开量子神经网络中的编码漏洞 quant-ph

7 Pages

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01486v1) [paper-pdf](http://arxiv.org/pdf/2502.01486v1)

**Authors**: Suryansh Upadhyay, Swaroop Ghosh

**Abstract**: Quantum computing (QC) has the potential to revolutionize fields like machine learning, security, and healthcare. Quantum machine learning (QML) has emerged as a promising area, enhancing learning algorithms using quantum computers. However, QML models are lucrative targets due to their high training costs and extensive training times. The scarcity of quantum resources and long wait times further exacerbate the challenge. Additionally, QML providers may rely on third party quantum clouds for hosting models, exposing them and their training data to potential threats. As QML as a Service (QMLaaS) becomes more prevalent, reliance on third party quantum clouds poses a significant security risk. This work demonstrates that adversaries in quantum cloud environments can exploit white box access to QML models to infer the users encoding scheme by analyzing circuit transpilation artifacts. The extracted data can be reused for training clone models or sold for profit. We validate the proposed attack through simulations, achieving high accuracy in distinguishing between encoding schemes. We report that 95% of the time, the encoding can be predicted correctly. To mitigate this threat, we propose a transient obfuscation layer that masks encoding fingerprints using randomized rotations and entanglement, reducing adversarial detection to near random chance 42% , with a depth overhead of 8.5% for a 5 layer QNN design.

摘要: 量子计算(QC)有可能给机器学习、安全和医疗保健等领域带来革命性的变化。量子机器学习(QML)已经成为一个很有前途的领域，它利用量子计算机来增强学习算法。然而，QML模型是有利可图的目标，因为它们的培训成本高，培训时间长。量子资源的稀缺和漫长的等待时间进一步加剧了挑战。此外，QML提供商可能会依赖第三方量子云来托管模型，从而使模型及其训练数据面临潜在威胁。随着QML即服务(QMLaaS)变得越来越普遍，对第三方量子云的依赖构成了重大的安全风险。这项工作表明，量子云环境中的攻击者可以利用白盒访问QML模型，通过分析电路转移伪影来推断用户的编码方案。提取的数据可以重复用于训练克隆模型或出售以赚取利润。我们通过仿真验证了所提出的攻击，在区分不同编码方案时达到了很高的准确率。我们报告说，95%的时间，编码可以被正确预测。为了缓解这种威胁，我们提出了一种瞬时混淆层，它使用随机旋转和纠缠来掩盖编码指纹，将敌意检测的概率降低到接近随机的42%，对于5层QNN设计，深度开销为8.5%。



## **14. DeTrigger: A Gradient-Centric Approach to Backdoor Attack Mitigation in Federated Learning**

DeTrigger：联邦学习中以用户为中心的后门攻击缓解方法 cs.LG

21 pages

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2411.12220v2) [paper-pdf](http://arxiv.org/pdf/2411.12220v2)

**Authors**: Kichang Lee, Yujin Shin, Jonghyuk Yun, Songkuk Kim, Jun Han, JeongGil Ko

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed devices while preserving local data privacy, making it ideal for mobile and embedded systems. However, the decentralized nature of FL also opens vulnerabilities to model poisoning attacks, particularly backdoor attacks, where adversaries implant trigger patterns to manipulate model predictions. In this paper, we propose DeTrigger, a scalable and efficient backdoor-robust federated learning framework that leverages insights from adversarial attack methodologies. By employing gradient analysis with temperature scaling, DeTrigger detects and isolates backdoor triggers, allowing for precise model weight pruning of backdoor activations without sacrificing benign model knowledge. Extensive evaluations across four widely used datasets demonstrate that DeTrigger achieves up to 251x faster detection than traditional methods and mitigates backdoor attacks by up to 98.9%, with minimal impact on global model accuracy. Our findings establish DeTrigger as a robust and scalable solution to protect federated learning environments against sophisticated backdoor threats.

摘要: 联合学习(FL)支持跨分布式设备进行协作模型培训，同时保护本地数据隐私，使其成为移动和嵌入式系统的理想选择。然而，FL的分散性也为建模中毒攻击打开了漏洞，特别是后门攻击，对手植入触发模式来操纵模型预测。在本文中，我们提出了DeTrigger，一个可扩展的高效后门健壮的联邦学习框架，它利用了对手攻击方法的见解。通过使用带有温度缩放的梯度分析，DeTrigger检测并隔离后门触发器，从而在不牺牲良性模型知识的情况下精确削减后门激活的模型权重。对四个广泛使用的数据集的广泛评估表明，DeTrigger的检测速度比传统方法快251倍，后门攻击减少高达98.9%，对全局模型精度的影响最小。我们的发现将DeTrigger确立为一个强大且可扩展的解决方案，可以保护联合学习环境免受复杂的后门威胁。



## **15. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

Top-FlipRAG：针对检索增强生成模型的面向主题的对抗性观点操纵攻击 cs.CL

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01386v1) [paper-pdf](http://arxiv.org/pdf/2502.01386v1)

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.

摘要: 基于大型语言模型(LLMS)的检索-增强生成(RAG)系统已经成为诸如问题回答和内容生成等任务的关键。然而，由于其固有的脆弱性，它们对舆论和信息传播的影响越来越大，使其成为安全研究的关键焦点。以前的研究主要针对针对事实或单一查询操作的攻击。在本文中，我们讨论了一个更实际的场景：针对RAG模型的面向主题的对抗性意见操纵攻击，其中要求LLM推理和综合多个视角，使它们特别容易受到系统性知识中毒的影响。具体地说，我们提出了Theme-FlipRAG，这是一种两阶段操纵攻击管道，战略性地制造对抗性扰动来影响相关查询的观点。该方法结合了传统的对抗性排序攻击技术，并利用LLMS丰富的内部相关知识和推理能力来执行语义级的扰动。实验表明，提出的攻击有效地改变了模型输出对特定主题的看法，显著影响了用户对信息的感知。目前的缓解方法无法有效防御此类攻击，这突显了加强RAG系统安全保障的必要性，并为LLM安全研究提供了至关重要的见解。



## **16. Detecting Backdoor Samples in Contrastive Language Image Pretraining**

对比语言图像预训练中后门样本检测 cs.LG

ICLR2025

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01385v1) [paper-pdf](http://arxiv.org/pdf/2502.01385v1)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: Contrastive language-image pretraining (CLIP) has been found to be vulnerable to poisoning backdoor attacks where the adversary can achieve an almost perfect attack success rate on CLIP models by poisoning only 0.01\% of the training dataset. This raises security concerns on the current practice of pretraining large-scale models on unscrutinized web data using CLIP. In this work, we analyze the representations of backdoor-poisoned samples learned by CLIP models and find that they exhibit unique characteristics in their local subspace, i.e., their local neighborhoods are far more sparse than that of clean samples. Based on this finding, we conduct a systematic study on detecting CLIP backdoor attacks and show that these attacks can be easily and efficiently detected by traditional density ratio-based local outlier detectors, whereas existing backdoor sample detection methods fail. Our experiments also reveal that an unintentional backdoor already exists in the original CC3M dataset and has been trained into a popular open-source model released by OpenCLIP. Based on our detector, one can clean up a million-scale web dataset (e.g., CC3M) efficiently within 15 minutes using 4 Nvidia A100 GPUs. The code is publicly available in our \href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub repository}.

摘要: 对比语言图像预训练(CLIP)被发现容易受到中毒后门攻击，对手只需中毒0.01%的训练数据集就可以在CLIP模型上获得近乎完美的攻击成功率。这引发了人们对当前使用CLIP对大规模模型进行未经审查的网络数据的预培训的安全担忧。在这项工作中，我们分析了通过CLIP模型学习的后门中毒样本的表示，发现它们在局部子空间中表现出独特的特征，即它们的局部邻域比干净样本的局部邻域稀疏得多。基于这一发现，我们对CLIP后门攻击的检测进行了系统的研究，结果表明，传统的基于密度比的局部离群点检测方法可以轻松有效地检测到这些攻击，而现有的后门样本检测方法却无法检测到这些攻击。我们的实验还表明，在原始的CC3M数据集中已经存在一个无意的后门，并且已经被训练成OpenCLIP发布的流行的开源模型。基于我们的检测器，您可以使用4个NVIDIA A100图形处理器在15分钟内高效清理百万级Web数据集(例如CC3M)。代码在我们的\href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub存储库中公开提供。



## **17. Improving the Robustness of Representation Misdirection for Large Language Model Unlearning**

提高大型语言模型去学习的表示误导的鲁棒性 cs.CL

12 pages, 4 figures, 1 table

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2501.19202v2) [paper-pdf](http://arxiv.org/pdf/2501.19202v2)

**Authors**: Dang Huu-Tien, Hoang Thanh-Tung, Le-Minh Nguyen, Naoya Inoue

**Abstract**: Representation Misdirection (RM) and variants are established large language model (LLM) unlearning methods with state-of-the-art performance. In this paper, we show that RM methods inherently reduce models' robustness, causing them to misbehave even when a single non-adversarial forget-token is in the retain-query. Toward understanding underlying causes, we reframe the unlearning process as backdoor attacks and defenses: forget-tokens act as backdoor triggers that, when activated in retain-queries, cause disruptions in RM models' behaviors, similar to successful backdoor attacks. To mitigate this vulnerability, we propose Random Noise Augmentation -- a model and method agnostic approach with theoretical guarantees for improving the robustness of RM methods. Extensive experiments demonstrate that RNA significantly improves the robustness of RM models while enhancing the unlearning performances.

摘要: 表示误导（RM）和变体是建立的大型语言模型（LLM）去学习方法，具有最先进的性能。在本文中，我们表明RM方法本质上降低了模型的鲁棒性，导致它们即使在保留查询中存在单个非对抗性遗忘令牌时也会表现不当。为了了解根本原因，我们将取消学习过程重新定义为后门攻击和防御：忘记令牌充当后门触发器，当在保留查询中激活时，会导致RM模型行为中断，类似于成功的后门攻击。为了减轻这一漏洞，我们提出了随机噪音增强--一种模型和方法不可知的方法，具有提高RM方法鲁棒性的理论保证。大量实验表明，RNA显着提高了RM模型的鲁棒性，同时增强了去学习性能。



## **18. Unified Breakdown Analysis for Byzantine Robust Gossip**

拜占庭稳健八卦的统一分解分析 math.OC

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2410.10418v2) [paper-pdf](http://arxiv.org/pdf/2410.10418v2)

**Authors**: Renaud Gaucher, Aymeric Dieuleveut, Hadrien Hendrikx

**Abstract**: In decentralized machine learning, different devices communicate in a peer-to-peer manner to collaboratively learn from each other's data. Such approaches are vulnerable to misbehaving (or Byzantine) devices. We introduce $\mathrm{F}\text{-}\rm RG$, a general framework for building robust decentralized algorithms with guarantees arising from robust-sum-like aggregation rules $\mathrm{F}$. We then investigate the notion of *breakdown point*, and show an upper bound on the number of adversaries that decentralized algorithms can tolerate. We introduce a practical robust aggregation rule, coined $\rm CS_{ours}$, such that $\rm CS_{ours}\text{-}RG$ has a near-optimal breakdown. Other choices of aggregation rules lead to existing algorithms such as $\rm ClippedGossip$ or $\rm NNA$. We give experimental evidence to validate the effectiveness of $\rm CS_{ours}\text{-}RG$ and highlight the gap with $\mathrm{NNA}$, in particular against a novel attack tailored to decentralized communications.

摘要: 在去中心化机器学习中，不同的设备以点对点的方式进行通信，以协作地从彼此的数据中学习。此类方法很容易受到行为不当（或拜占庭式）设备的影响。我们引入了$\mathrm{F}\text{-}\rm RG$，这是一个用于构建稳健去中心化算法的通用框架，其保证源自类似稳健和的聚合规则$\mathrm{F}$。然后我们研究 * 崩溃点 * 的概念，并给出去中心化算法可以容忍的对手数量的上限。我们引入了一个实用的鲁棒聚合规则，创造了$\rm CS_{our}$，以便$\rm CS_{our}\text{-}RG$具有近乎最优的细分。聚合规则的其他选择导致现有算法，例如$\rm ClipedGossip $或$\rm NNA$。我们提供了实验证据来验证$\rm CS_{our}\text{-}RG$的有效性，并强调了与$\mathrm{NNA}$的差距，特别是针对去中心化通信量身定制的新型攻击。



## **19. FSPGD: Rethinking Black-box Attacks on Semantic Segmentation**

FSPVD：重新思考对语义分割的黑匣子攻击 cs.CV

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01262v1) [paper-pdf](http://arxiv.org/pdf/2502.01262v1)

**Authors**: Eun-Sol Park, MiSo Park, Seung Park, Yong-Goo Shin

**Abstract**: Transferability, the ability of adversarial examples crafted for one model to deceive other models, is crucial for black-box attacks. Despite advancements in attack methods for semantic segmentation, transferability remains limited, reducing their effectiveness in real-world applications. To address this, we introduce the Feature Similarity Projected Gradient Descent (FSPGD) attack, a novel black-box approach that enhances both attack performance and transferability. Unlike conventional segmentation attacks that rely on output predictions for gradient calculation, FSPGD computes gradients from intermediate layer features. Specifically, our method introduces a loss function that targets local information by comparing features between clean images and adversarial examples, while also disrupting contextual information by accounting for spatial relationships between objects. Experiments on Pascal VOC 2012 and Cityscapes datasets demonstrate that FSPGD achieves superior transferability and attack performance, establishing a new state-of-the-art benchmark. Code is available at https://github.com/KU-AIVS/FSPGD.

摘要: 可转移性，即为一个模型制作的敌意例子欺骗其他模型的能力，对黑盒攻击至关重要。尽管语义分割的攻击方法有所进步，但可转移性仍然有限，降低了它们在现实世界应用中的有效性。为了解决这一问题，我们引入了特征相似性投影梯度下降(FSPGD)攻击，这是一种新的黑盒方法，它同时提高了攻击性能和可转移性。与依赖输出预测进行梯度计算的传统分割攻击不同，FSPGD根据中间层特征计算梯度。具体地说，我们的方法引入了一个损失函数，该函数通过比较干净图像和敌意图像之间的特征来定位局部信息，同时还通过考虑对象之间的空间关系来破坏上下文信息。在Pascal VOC 2012和CITYSCAPES数据集上的实验表明，FSPGD实现了卓越的可转移性和攻击性能，建立了一个新的最先进的基准。代码可在https://github.com/KU-AIVS/FSPGD.上找到



## **20. The Impact of Logic Locking on Confidentiality: An Automated Evaluation**

逻辑锁定对保密性的影响：自动评估 cs.CR

8 pages, accepted at 26th International Symposium on Quality  Electronic Design (ISQED'25)

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01240v1) [paper-pdf](http://arxiv.org/pdf/2502.01240v1)

**Authors**: Lennart M. Reimann, Evgenii Rezunov, Dominik Germek, Luca Collini, Christian Pilato, Ramesh Karri, Rainer Leupers

**Abstract**: Logic locking secures hardware designs in untrusted foundries by incorporating key-driven gates to obscure the original blueprint. While this method safeguards the integrated circuit from malicious alterations during fabrication, its influence on data confidentiality during runtime has been ignored. In this study, we employ path sensitization to formally examine the impact of logic locking on confidentiality. By applying three representative logic locking mechanisms on open-source cryptographic benchmarks, we utilize an automatic test pattern generation framework to evaluate the effect of locking on cryptographic encryption keys and sensitive data signals. Our analysis reveals that logic locking can inadvertently cause sensitive data leakage when incorrect logic locking keys are used. We show that a single malicious logic locking key can expose over 70% of an encryption key. If an adversary gains control over other inputs, the entire encryption key can be compromised. This research uncovers a significant security vulnerability in logic locking and emphasizes the need for comprehensive security assessments that extend beyond key-recovery attacks.

摘要: 逻辑锁定通过结合钥匙驱动的门来模糊原始蓝图，从而保护不可信铸造厂中的硬件设计。虽然这种方法保护集成电路在制造过程中不受恶意更改，但它对运行时数据保密性的影响被忽略。在这项研究中，我们使用路径敏感化来形式化地检查逻辑锁定对机密性的影响。通过将三种典型的逻辑锁定机制应用于开源密码基准测试，我们利用一个自动测试模式生成框架来评估锁定对加密密钥和敏感数据信号的影响。我们的分析表明，当使用了错误的逻辑锁密钥时，逻辑锁可能会无意中导致敏感数据泄漏。我们发现，一个恶意的逻辑锁密钥可以暴露超过70%的加密密钥。如果对手获得了对其他输入的控制，则整个加密密钥可能会被泄露。这项研究揭示了逻辑锁定中的一个重大安全漏洞，并强调需要进行全面的安全评估，而不仅仅是密钥恢复攻击。



## **21. The dark deep side of DeepSeek: Fine-tuning attacks against the safety alignment of CoT-enabled models**

DeepSeek的阴暗面：针对支持CoT的模型的安全一致性的微调攻击 cs.CR

12 Pages

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01225v1) [paper-pdf](http://arxiv.org/pdf/2502.01225v1)

**Authors**: Zhiyuan Xu, Joseph Gardiner, Sana Belguith

**Abstract**: Large language models are typically trained on vast amounts of data during the pre-training phase, which may include some potentially harmful information. Fine-tuning attacks can exploit this by prompting the model to reveal such behaviours, leading to the generation of harmful content. In this paper, we focus on investigating the performance of the Chain of Thought based reasoning model, DeepSeek, when subjected to fine-tuning attacks. Specifically, we explore how fine-tuning manipulates the model's output, exacerbating the harmfulness of its responses while examining the interaction between the Chain of Thought reasoning and adversarial inputs. Through this study, we aim to shed light on the vulnerability of Chain of Thought enabled models to fine-tuning attacks and the implications for their safety and ethical deployment.

摘要: 大型语言模型通常在预训练阶段根据大量数据进行训练，其中可能包括一些潜在有害的信息。微调攻击可以通过促使模型揭示此类行为来利用这一点，从而导致有害内容的生成。在本文中，我们重点研究基于思想链的推理模型DeepSeek在受到微调攻击时的性能。具体来说，我们探索微调如何操纵模型的输出，加剧其反应的危害性，同时检查思维链推理和对抗输入之间的相互作用。通过这项研究，我们的目标是揭示思想链使模型能够微调攻击的脆弱性及其对安全性和道德部署的影响。



## **22. Jailbreaking with Universal Multi-Prompts**

用通用多胞胎越狱 cs.CL

Accepted by NAACL Findings 2025

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01154v1) [paper-pdf](http://arxiv.org/pdf/2502.01154v1)

**Authors**: Yu-Ling Hsu, Hsuan Su, Shang-Tse Chen

**Abstract**: Large language models (LLMs) have seen rapid development in recent years, revolutionizing various applications and significantly enhancing convenience and productivity. However, alongside their impressive capabilities, ethical concerns and new types of attacks, such as jailbreaking, have emerged. While most prompting techniques focus on optimizing adversarial inputs for individual cases, resulting in higher computational costs when dealing with large datasets. Less research has addressed the more general setting of training a universal attacker that can transfer to unseen tasks. In this paper, we introduce JUMP, a prompt-based method designed to jailbreak LLMs using universal multi-prompts. We also adapt our approach for defense, which we term DUMP. Experimental results demonstrate that our method for optimizing universal multi-prompts outperforms existing techniques.

摘要: 近年来，大型语言模型（LLM）发展迅速，彻底改变了各种应用程序，显着提高了便利性和生产力。然而，除了它们令人印象深刻的能力之外，道德问题和越狱等新型攻击也出现了。虽然大多数提示技术专注于优化个别案例的对抗输入，从而导致处理大型数据集时计算成本更高。较少的研究涉及训练可以转移到不可见任务的通用攻击者的更一般设置。本文中，我们介绍了JUMP，这是一种基于预算的方法，旨在使用通用多提示越狱LLM。我们还调整我们的防御方法，我们称之为“DUMP”。实验结果表明，我们用于优化通用多提示的方法优于现有技术。



## **23. Warfare:Breaking the Watermark Protection of AI-Generated Content**

战争：打破人工智能生成内容的水印保护 cs.CV

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2310.07726v4) [paper-pdf](http://arxiv.org/pdf/2310.07726v4)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Shangwei Guo, Han Qiu, Guoyin Wang, Jiwei Li, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is rapidly expanding, with services using advanced generative models to create realistic images and fluent text. Regulating such content is crucial to prevent policy violations, such as unauthorized commercialization or unsafe content distribution. Watermarking is a promising solution for content attribution and verification, but we demonstrate its vulnerability to two key attacks: (1) Watermark removal, where adversaries erase embedded marks to evade regulation, and (2) Watermark forging, where they generate illicit content with forged watermarks, leading to misattribution. We propose Warfare, a unified attack framework leveraging a pre-trained diffusion model for content processing and a generative adversarial network for watermark manipulation. Evaluations across datasets and embedding setups show that Warfare achieves high success rates while preserving content quality. We further introduce Warfare-Plus, which enhances efficiency without compromising effectiveness. The code can be found in https://github.com/GuanlinLee/warfare.

摘要: 人工智能生成的内容(AIGC)正在迅速扩展，服务使用先进的生成模型来创建逼真的图像和流畅的文本。监管此类内容对于防止违反政策至关重要，例如未经授权的商业化或不安全的内容分发。水印是一种很有前途的内容归属和验证解决方案，但我们证明了它对两个关键攻击的脆弱性：(1)水印移除，攻击者删除嵌入的标记以逃避监管；(2)水印伪造，他们生成含有伪造水印的非法内容，导致错误归属。我们提出了一种统一的攻击框架，该框架利用预先训练的扩散模型进行内容处理，并利用生成性对抗网络来处理水印。对数据集和嵌入设置进行的评估表明，WARFARE在保持内容质量的同时实现了高成功率。我们进一步引入了Warfare-Plus，它在不影响有效性的情况下提高了效率。代码可以在https://github.com/GuanlinLee/warfare.中找到



## **24. Adversarial Robustness in Two-Stage Learning-to-Defer: Algorithms and Guarantees**

两阶段学习推迟中的对抗稳健性：算法和保证 stat.ML

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01027v1) [paper-pdf](http://arxiv.org/pdf/2502.01027v1)

**Authors**: Yannis Montreuil, Axel Carlier, Lai Xing Ng, Wei Tsang Ooi

**Abstract**: Learning-to-Defer (L2D) facilitates optimal task allocation between AI systems and decision-makers. Despite its potential, we show that current two-stage L2D frameworks are highly vulnerable to adversarial attacks, which can misdirect queries or overwhelm decision agents, significantly degrading system performance. This paper conducts the first comprehensive analysis of adversarial robustness in two-stage L2D frameworks. We introduce two novel attack strategies -- untargeted and targeted -- that exploit inherent structural vulnerabilities in these systems. To mitigate these threats, we propose SARD, a robust, convex, deferral algorithm rooted in Bayes and $(\mathcal{R},\mathcal{G})$-consistency. Our approach guarantees optimal task allocation under adversarial perturbations for all surrogates in the cross-entropy family. Extensive experiments on classification, regression, and multi-task benchmarks validate the robustness of SARD.

摘要: 学习延迟（L2 D）促进人工智能系统和决策者之间的最佳任务分配。尽管有潜力，但我们表明当前的两阶段L2 D框架非常容易受到对抗攻击，这些攻击可能会误导查询或压倒决策代理，从而显着降低系统性能。本文首次对两阶段L2 D框架中的对抗鲁棒性进行了全面分析。我们引入了两种新颖的攻击策略--无针对性和有针对性--它们利用这些系统中固有的结构性漏洞。为了减轻这些威胁，我们提出了SAARD，这是一种基于Bayes和$（\mathcal{R}，\mathcal{G}）$-一致性的稳健、凸、延迟算法。我们的方法保证了交叉熵家族中所有代理人在对抗扰动下的最佳任务分配。关于分类、回归和多任务基准的大量实验验证了SARD的稳健性。



## **25. Refining Adaptive Zeroth-Order Optimization at Ease**

轻松细化自适应零阶优化 cs.LG

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01014v1) [paper-pdf](http://arxiv.org/pdf/2502.01014v1)

**Authors**: Yao Shu, Qixin Zhang, Kun He, Zhongxiang Dai

**Abstract**: Recently, zeroth-order (ZO) optimization plays an essential role in scenarios where gradient information is inaccessible or unaffordable, such as black-box systems and resource-constrained environments. While existing adaptive methods such as ZO-AdaMM have shown promise, they are fundamentally limited by their underutilization of moment information during optimization, usually resulting in underperforming convergence. To overcome these limitations, this paper introduces Refined Adaptive Zeroth-Order Optimization (R-AdaZO). Specifically, we first show the untapped variance reduction effect of first moment estimate on ZO gradient estimation, which improves the accuracy and stability of ZO updates. We then refine the second moment estimate based on these variance-reduced gradient estimates to better capture the geometry of the optimization landscape, enabling a more effective scaling of ZO updates. We present rigorous theoretical analysis to show (I) the first analysis to the variance reduction of first moment estimate in ZO optimization, (II) the improved second moment estimates with a more accurate approximation of its variance-free ideal, (III) the first variance-aware convergence framework for adaptive ZO methods, which may be of independent interest, and (IV) the faster convergence of R-AdaZO than existing baselines like ZO-AdaMM. Our extensive experiments, including synthetic problems, black-box adversarial attack, and memory-efficient fine-tuning of large language models (LLMs), further verify the superior convergence of R-AdaZO, indicating that R-AdaZO offers an improved solution for real-world ZO optimization challenges.

摘要: 最近，零阶(ZO)优化在诸如黑盒系统和资源受限环境等无法获取或负担不起梯度信息的场景中扮演着重要的角色。虽然现有的自适应方法如ZO-AdaMM已经显示出很好的前景，但它们在优化过程中对矩信息的利用不足从根本上限制了它们，通常导致收敛性能不佳。为了克服这些局限性，本文引入了改进的自适应零阶优化算法(R-AdaZO)。具体地说，我们首先展示了一阶矩估计对ZO梯度估计的未开发的减方差效果，从而提高了ZO更新的精度和稳定性。然后，我们基于这些经方差减少的梯度估计来改进二阶矩估计，以更好地捕捉优化场景的几何形状，从而实现更有效的ZO更新缩放。我们给出了严格的理论分析，以证明(I)第一次分析ZO优化中一阶矩估计的方差降低，(Ii)改进的二阶矩估计更精确地逼近其无方差理想，(Iii)自适应ZO方法的第一个方差感知收敛框架，它可能是独立的，以及(Iv)R-AdaZO比现有基线(如ZO-AdaMM)更快的收敛。我们的大量实验，包括合成问题、黑盒对抗攻击和对大型语言模型(LLM)的内存效率优化，进一步验证了R-AdaZO的优越收敛能力，表明R-AdaZO为现实世界的ZO优化挑战提供了一种改进的解决方案。



## **26. Boosting Adversarial Robustness and Generalization with Structural Prior**

利用结构先验增强对抗稳健性和推广 cs.LG

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00834v1) [paper-pdf](http://arxiv.org/pdf/2502.00834v1)

**Authors**: Zhichao Hou, Weizhi Gao, Hamid Krim, Xiaorui Liu

**Abstract**: This work investigates a novel approach to boost adversarial robustness and generalization by incorporating structural prior into the design of deep learning models. Specifically, our study surprisingly reveals that existing dictionary learning-inspired convolutional neural networks (CNNs) provide a false sense of security against adversarial attacks. To address this, we propose Elastic Dictionary Learning Networks (EDLNets), a novel ResNet architecture that significantly enhances adversarial robustness and generalization. This novel and effective approach is supported by a theoretical robustness analysis using influence functions. Moreover, extensive and reliable experiments demonstrate consistent and significant performance improvement on open robustness leaderboards such as RobustBench, surpassing state-of-the-art baselines. To the best of our knowledge, this is the first work to discover and validate that structural prior can reliably enhance deep learning robustness under strong adaptive attacks, unveiling a promising direction for future research.

摘要: 这项工作研究了一种新的方法，通过将结构先验引入深度学习模型的设计中来提高对手健壮性和泛化能力。具体地说，我们的研究令人惊讶地发现，现有的受词典学习启发的卷积神经网络(CNN)针对对手攻击提供了一种错误的安全感。为了解决这一问题，我们提出了弹性词典学习网络(EDLNets)，这是一种新的ResNet体系结构，显著增强了对手的健壮性和泛化能力。这一新颖而有效的方法得到了使用影响函数的理论稳健性分析的支持。此外，广泛和可靠的实验表明，在开放的健壮性排行榜上，如RobustBch，性能得到了一致和显著的改善，超过了最先进的基线。据我们所知，这是第一次发现和验证结构先验在强自适应攻击下能够可靠地增强深度学习的稳健性，为未来的研究提供了一个很有前途的方向。



## **27. AGNNCert: Defending Graph Neural Networks against Arbitrary Perturbations with Deterministic Certification**

AGNNCert：通过确定性认证保护图神经网络免受任意扰动 cs.CR

Accepted by Usenix Security 2025

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00765v1) [paper-pdf](http://arxiv.org/pdf/2502.00765v1)

**Authors**: Jiate Li, Binghui Wang

**Abstract**: Graph neural networks (GNNs) achieve the state-of-the-art on graph-relevant tasks such as node and graph classification. However, recent works show GNNs are vulnerable to adversarial perturbations include the perturbation on edges, nodes, and node features, the three components forming a graph. Empirical defenses against such attacks are soon broken by adaptive ones. While certified defenses offer robustness guarantees, they face several limitations: 1) almost all restrict the adversary's capability to only one type of perturbation, which is impractical; 2) all are designed for a particular GNN task, which limits their applicability; and 3) the robustness guarantees of all methods except one are not 100% accurate.   We address all these limitations by developing AGNNCert, the first certified defense for GNNs against arbitrary (edge, node, and node feature) perturbations with deterministic robustness guarantees, and applicable to the two most common node and graph classification tasks. AGNNCert also encompass existing certified defenses as special cases. Extensive evaluations on multiple benchmark node/graph classification datasets and two real-world graph datasets, and multiple GNNs validate the effectiveness of AGNNCert to provably defend against arbitrary perturbations. AGNNCert also shows its superiority over the state-of-the-art certified defenses against the individual edge perturbation and node perturbation.

摘要: 图神经网络(GNN)在节点和图分类等与图相关的任务上实现了最先进的技术。然而，最近的研究表明，GNN容易受到对抗性扰动的影响，包括边、节点和节点特征的扰动，这三个组成部分构成了一个图。针对此类攻击的经验防御很快就会被适应性防御所打破。虽然认证防御提供了健壮性保证，但它们面临着几个限制：1)几乎所有方法都将对手的能力限制为只有一种类型的扰动，这是不切实际的；2)所有方法都是为特定的GNN任务设计的，这限制了它们的适用性；以及3)除一种方法外，所有方法的健壮性保证都不是100%准确的。我们通过开发AGNNCert来解决所有这些限制，AGNNCert是第一个经过认证的GNN防御任意(边、节点和节点特征)扰动的方法，具有确定性的健壮性保证，适用于两个最常见的节点和图分类任务。AGNNCert还将现有的经认证的辩护作为特例包括在内。在多个基准节点/图分类数据集和两个真实世界图数据集以及多个GNN上的广泛评估验证了AGNNCert在抵御任意扰动方面的有效性。AGNNCert还显示了其相对于最先进的认证防御系统的优越性，以抵御个人边缘扰动和节点扰动。



## **28. Decentralized Nonconvex Robust Optimization over Unsafe Multiagent Systems: System Modeling, Utility, Resilience, and Privacy Analysis**

不安全多智能体系统上的分散非凸鲁棒优化：系统建模、效用、弹性和隐私分析 math.OC

15 pages, 15 figures

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2409.18632v6) [paper-pdf](http://arxiv.org/pdf/2409.18632v6)

**Authors**: Jinhui Hu, Guo Chen, Huaqing Li, Huqiang Cheng, Xiaoyu Guo, Tingwen Huang

**Abstract**: Privacy leakage and Byzantine failures are two adverse factors to the intelligent decision-making process of multi-agent systems (MASs). Considering the presence of these two issues, this paper targets the resolution of a class of nonconvex optimization problems under the Polyak-{\L}ojasiewicz (P-{\L}) condition. To address this problem, we first identify and construct the adversary system model. To enhance the robustness of stochastic gradient descent methods, we mask the local gradients with Gaussian noises and adopt a resilient aggregation method self-centered clipping (SCC) to design a differentially private (DP) decentralized Byzantine-resilient algorithm, namely DP-SCC-PL, which simultaneously achieves differential privacy and Byzantine resilience. The convergence analysis of DP-SCC-PL is challenging since the convergence error can be contributed jointly by privacy-preserving and Byzantine-resilient mechanisms, as well as the nonconvex relaxation, which is addressed via seeking the contraction relationships among the disagreement measure of reliable agents before and after aggregation, together with the optimal gap. Theoretical results reveal that DP-SCC-PL achieves consensus among all reliable agents and sublinear (inexact) convergence with well-designed step-sizes. It has also been proved that if there are no privacy issues and Byzantine agents, then the asymptotic exact convergence can be recovered. Numerical experiments verify the utility, resilience, and differential privacy of DP-SCC-PL by tackling a nonconvex optimization problem satisfying the P-{\L} condition under various Byzantine attacks.

摘要: 隐私泄露和拜占庭失效是影响多智能体系统(MASS)智能决策过程的两个不利因素。考虑到这两个问题的存在，本文研究了一类在Polyak-L条件下的非凸优化问题的解。为了解决这一问题，我们首先识别并构建了对手系统模型。为了增强随机梯度下降算法的稳健性，我们用高斯噪声掩盖局部梯度，并采用弹性聚合方法自中心剪裁(SCC)设计了一种差分私有(DP)分散拜占庭弹性算法DP-SCC-PL，同时实现了差分隐私保护和拜占庭弹性。DP-SCC-PL的收敛分析具有挑战性，因为收敛误差是由隐私保护和拜占庭弹性机制以及非凸松弛机制共同造成的，非凸松弛通过寻找可靠代理聚集前后的不一致度量和最优间隙之间的收缩关系来解决。理论结果表明，DP-SCC-PL算法在合理设计步长的情况下，实现了所有可靠代理之间的一致性和次线性(不精确)收敛。证明了如果不存在隐私问题和拜占庭代理，则可以恢复渐近精确收敛。通过求解满足P-L条件的非凸优化问题，验证了DP-SCC-PL在不同拜占庭攻击下的实用性、抗攻击能力和差分隐私性。



## **29. Gandalf the Red: Adaptive Security for LLMs**

红色甘道夫：LLM的自适应安全 cs.LG

Niklas Pfister, V\'aclav Volhejn and Manuel Knott contributed equally

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2501.07927v2) [paper-pdf](http://arxiv.org/pdf/2501.07927v2)

**Authors**: Niklas Pfister, Václav Volhejn, Manuel Knott, Santiago Arias, Julia Bazińska, Mykhailo Bichurin, Alan Commike, Janet Darling, Peter Dienes, Matthew Fiedler, David Haber, Matthias Kraft, Marco Lancini, Max Mathys, Damián Pascual-Ortiz, Jakub Podolak, Adrià Romero-López, Kyriacos Shiarlis, Andreas Signer, Zsolt Terek, Athanasios Theocharis, Daniel Timbrell, Samuel Trautwein, Samuel Watts, Yun-Han Wu, Mateo Rojas-Carulla

**Abstract**: Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications.

摘要: 目前对大型语言模型(LLM)应用程序中针对即时攻击的防御措施的评估往往忽略了两个关键因素：敌意行为的动态性质和限制性防御措施对合法用户的可用性惩罚。本文提出了动态安全效用威胁模型D-SEC，它明确地将攻击者和合法用户分开，对多步交互进行建模，并以优化的形式表达安全效用。我们通过引入Gandalf进一步解决了现有评估中的缺陷，Gandalf是一个众包、游戏化的红色团队平台，旨在生成现实的、自适应的攻击。使用Gandalf，我们收集并发布了279K提示攻击的数据集。在良性用户数据的补充下，我们的分析揭示了安全性和实用性之间的相互作用，表明LLM中集成的防御措施(例如系统提示)即使在不阻止请求的情况下也会降低可用性。我们演示了受限应用程序域、深度防御和自适应防御是构建安全且有用的LLM应用程序的有效策略。



## **30. From Compliance to Exploitation: Jailbreak Prompt Attacks on Multimodal LLMs**

从合规到剥削：对多模式LLM的越狱立即攻击 cs.CR

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00735v1) [paper-pdf](http://arxiv.org/pdf/2502.00735v1)

**Authors**: Chun Wai Chiu, Linghan Huang, Bo Li, Huaming Chen

**Abstract**: Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the frontier multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. To better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flank Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios. These findings highlight both the potency of prompt-based obfuscation in voice-enabled contexts and the limitations of current LLMs' moderation safeguards and the urgent need for advanced defense strategies to address the challenges posed by evolving, context-rich attacks.

摘要: 大语言模型因其处理文本、音频、图像和视频等不同类型输入数据的能力日益增强，在各个领域得到了广泛的应用。虽然LLM在理解和生成不同场景的上下文方面表现出了出色的性能，但它们很容易受到基于提示的攻击，这些攻击主要是通过文本输入进行的。在本文中，我们介绍了第一个基于语音的针对多模式LLMS的越狱攻击，称为侧翼攻击，它可以同时处理针对多模式LLMS的不同类型的输入。我们的工作是受到单语言语音驱动的大型语言模型的最新进展的推动，这些模型在传统的基于文本的LLMS漏洞之外引入了新的攻击面。为了调查这些风险，我们研究了前沿多模式LLMS，这些LLMS可以通过不同类型的输入(如音频输入)访问，重点关注对抗性提示如何绕过其防御机制。我们提出了一种新的策略，在不允许的提示的两侧是良性的、叙事驱动的提示。它被整合到侧翼攻击中，试图使交互上下文人性化，并通过虚构的设置执行攻击。为了更好地评估攻击性能，我们提出了一个半自动的策略违规检测自我评估框架。我们证明了侧翼攻击能够操纵最先进的LLM产生未对齐和禁止的输出，在七个禁止场景中获得了从0.67到0.93的平均攻击成功率。这些发现既突显了基于提示的混淆在语音支持的上下文中的有效性，也突显了当前LLMS适度保障的局限性，以及迫切需要先进的防御策略来应对不断演变的、上下文丰富的攻击带来的挑战。



## **31. "I am bad": Interpreting Stealthy, Universal and Robust Audio Jailbreaks in Audio-Language Models**

“我很坏”：在音频语言模型中解释秘密、普遍和稳健的音频越狱 cs.LG

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00718v1) [paper-pdf](http://arxiv.org/pdf/2502.00718v1)

**Authors**: Isha Gupta, David Khachaturov, Robert Mullins

**Abstract**: The rise of multimodal large language models has introduced innovative human-machine interaction paradigms but also significant challenges in machine learning safety. Audio-Language Models (ALMs) are especially relevant due to the intuitive nature of spoken communication, yet little is known about their failure modes. This paper explores audio jailbreaks targeting ALMs, focusing on their ability to bypass alignment mechanisms. We construct adversarial perturbations that generalize across prompts, tasks, and even base audio samples, demonstrating the first universal jailbreaks in the audio modality, and show that these remain effective in simulated real-world conditions. Beyond demonstrating attack feasibility, we analyze how ALMs interpret these audio adversarial examples and reveal them to encode imperceptible first-person toxic speech - suggesting that the most effective perturbations for eliciting toxic outputs specifically embed linguistic features within the audio signal. These results have important implications for understanding the interactions between different modalities in multimodal models, and offer actionable insights for enhancing defenses against adversarial audio attacks.

摘要: 多通道大型语言模型的兴起引入了创新的人机交互范式，但也给机器学习的安全性带来了重大挑战。由于口语交流的直觉性，音频语言模型(ALM)尤其相关，但人们对其失败模式知之甚少。本文探讨了针对施舍的音频越狱，重点是它们绕过对齐机制的能力。我们构建了跨提示、任务甚至基本音频样本的对抗性扰动，演示了音频通道中的第一个通用越狱，并表明这些扰动在模拟的真实世界条件下仍然有效。除了展示攻击的可行性外，我们还分析了ALMS如何解释这些音频对抗性例子，并将它们揭示为编码不可感知的第一人称有毒言语--这表明，引发有毒输出的最有效扰动具体地将语言特征嵌入音频信号中。这些结果对于理解多通道模型中不同通道之间的相互作用具有重要意义，并为增强对敌方音频攻击的防御提供了可操作的见解。



## **32. Transferable Adversarial Face Attack with Text Controlled Attribute**

具有文本控制属性的可转移对抗面部攻击 cs.CV

Accepted by AAAI 2025

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2412.11735v2) [paper-pdf](http://arxiv.org/pdf/2412.11735v2)

**Authors**: Wenyun Li, Zheng Zhang, Xiangyuan Lan, Dongmei Jiang

**Abstract**: Traditional adversarial attacks typically produce adversarial examples under norm-constrained conditions, whereas unrestricted adversarial examples are free-form with semantically meaningful perturbations. Current unrestricted adversarial impersonation attacks exhibit limited control over adversarial face attributes and often suffer from low transferability. In this paper, we propose a novel Text Controlled Attribute Attack (TCA$^2$) to generate photorealistic adversarial impersonation faces guided by natural language. Specifically, the category-level personal softmax vector is employed to precisely guide the impersonation attacks. Additionally, we propose both data and model augmentation strategies to achieve transferable attacks on unknown target models. Finally, a generative model, \textit{i.e}, Style-GAN, is utilized to synthesize impersonated faces with desired attributes. Extensive experiments on two high-resolution face recognition datasets validate that our TCA$^2$ method can generate natural text-guided adversarial impersonation faces with high transferability. We also evaluate our method on real-world face recognition systems, \textit{i.e}, Face++ and Aliyun, further demonstrating the practical potential of our approach.

摘要: 传统的对抗性攻击通常在范数受限的条件下产生对抗性示例，而不受限制的对抗性示例是自由形式的，具有语义意义的扰动。当前不受限制的对抗性模仿攻击对对抗性面孔属性的控制有限，并且往往存在可转移性低的问题。本文提出了一种新的文本控制属性攻击(TCA$^2$)，用于生成自然语言引导下的照片真实感对抗性模仿人脸。具体地说，采用类别级的个人Softmax向量来精确地指导模仿攻击。此外，我们还提出了数据和模型扩充策略来实现对未知目标模型的可转移攻击。最后，利用一个生成模型在两个高分辨率人脸识别数据集上的大量实验验证了我们的TCA$^2$方法能够生成具有很高可转移性的自然文本引导的对抗性模拟人脸。我们还在真实的人脸识别系统



## **33. The Great Contradiction Showdown: How Jailbreak and Stealth Wrestle in Vision-Language Models?**

巨大的矛盾对决：越狱和隐形如何在视觉语言模型中摔跤？ cs.LG

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2410.01438v2) [paper-pdf](http://arxiv.org/pdf/2410.01438v2)

**Authors**: Ching-Chia Kao, Chia-Mu Yu, Chun-Shien Lu, Chu-Song Chen

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable performance on a variety of tasks, yet they remain vulnerable to jailbreak attacks that compromise safety and reliability. In this paper, we provide an information-theoretic framework for understanding the fundamental trade-off between the effectiveness of these attacks and their stealthiness. Drawing on Fano's inequality, we demonstrate how an attacker's success probability is intrinsically linked to the stealthiness of generated prompts. Building on this, we propose an efficient algorithm for detecting non-stealthy jailbreak attacks, offering significant improvements in model robustness. Experimental results highlight the tension between strong attacks and their detectability, providing insights into both adversarial strategies and defense mechanisms.

摘要: 视觉语言模型（VLM）在各种任务中取得了出色的性能，但它们仍然容易受到危及安全性和可靠性的越狱攻击。在本文中，我们提供了一个信息论框架来理解这些攻击的有效性与其隐蔽性之间的基本权衡。利用法诺的不等式，我们展示了攻击者的成功概率如何与生成提示的隐秘性存在内在联系。在此基础上，我们提出了一种用于检测非隐形越狱攻击的高效算法，在模型稳健性方面提供了显着改进。实验结果凸显了强攻击及其可检测性之间的紧张关系，为对抗策略和防御机制提供了见解。



## **34. Towards Robust Multimodal Large Language Models Against Jailbreak Attacks**

迈向抵御越狱攻击的稳健多模式大型语言模型 cs.CR

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00653v1) [paper-pdf](http://arxiv.org/pdf/2502.00653v1)

**Authors**: Ziyi Yin, Yuanpu Cao, Han Liu, Ting Wang, Jinghui Chen, Fenhlong Ma

**Abstract**: While multimodal large language models (MLLMs) have achieved remarkable success in recent advancements, their susceptibility to jailbreak attacks has come to light. In such attacks, adversaries exploit carefully crafted prompts to coerce models into generating harmful or undesirable content. Existing defense mechanisms often rely on external inference steps or safety alignment training, both of which are less effective and impractical when facing sophisticated adversarial perturbations in white-box scenarios. To address these challenges and bolster MLLM robustness, we introduce SafeMLLM by adopting an adversarial training framework that alternates between an attack step for generating adversarial noise and a model updating step. At the attack step, SafeMLLM generates adversarial perturbations through a newly proposed contrastive embedding attack (CoE-Attack), which optimizes token embeddings under a contrastive objective. SafeMLLM then updates model parameters to neutralize the perturbation effects while preserving model utility on benign inputs. We evaluate SafeMLLM across six MLLMs and six jailbreak methods spanning multiple modalities. Experimental results show that SafeMLLM effectively defends against diverse attacks, maintaining robust performance and utilities.

摘要: 虽然多模式大型语言模型(MLLM)在最近的进步中取得了显著的成功，但它们对越狱攻击的敏感性已经暴露出来。在此类攻击中，攻击者利用精心设计的提示来强迫模型生成有害或不受欢迎的内容。现有的防御机制往往依赖于外部推理步骤或安全对齐训练，在白盒场景中面对复杂的对手扰动时，这两种方法都不太有效和不切实际。为了应对这些挑战并增强MLLM的稳健性，我们引入了SafeMLLM，采用了一种对抗性训练框架，该框架在生成对抗性噪声的攻击步骤和模型更新步骤之间交替。在攻击阶段，SafeMLLM通过新提出的对比性嵌入攻击(COE-Attack)来产生敌意扰动，该攻击在对比性目标下优化令牌嵌入。SafeMLLM然后更新模型参数，以中和扰动影响，同时保留对良性输入的模型效用。我们评估了六种MLLM和六种越狱方法的SafeMLLM，这些方法跨越多个医疗设备。实验结果表明，SafeMLLM能够有效地防御各种攻击，并保持了较强的性能和实用性。



## **35. Reformulation is All You Need: Addressing Malicious Text Features in DNNs**

重新制定即可：解决DNN中的恶意文本特征 cs.LG

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00652v1) [paper-pdf](http://arxiv.org/pdf/2502.00652v1)

**Authors**: Yi Jiang, Oubo Ma, Yong Yang, Tong Zhang, Shouling Ji

**Abstract**: Human language encompasses a wide range of intricate and diverse implicit features, which attackers can exploit to launch adversarial or backdoor attacks, compromising DNN models for NLP tasks. Existing model-oriented defenses often require substantial computational resources as model size increases, whereas sample-oriented defenses typically focus on specific attack vectors or schemes, rendering them vulnerable to adaptive attacks. We observe that the root cause of both adversarial and backdoor attacks lies in the encoding process of DNN models, where subtle textual features, negligible for human comprehension, are erroneously assigned significant weight by less robust or trojaned models. Based on it we propose a unified and adaptive defense framework that is effective against both adversarial and backdoor attacks. Our approach leverages reformulation modules to address potential malicious features in textual inputs while preserving the original semantic integrity. Extensive experiments demonstrate that our framework outperforms existing sample-oriented defense baselines across a diverse range of malicious textual features.

摘要: 人类语言包含一系列复杂多样的隐含功能，攻击者可以利用这些功能发动对抗性攻击或后门攻击，从而危及NLP任务的DNN模型。随着模型大小的增加，现有的面向模型的防御通常需要大量的计算资源，而面向样本的防御通常专注于特定的攻击向量或方案，从而使它们容易受到自适应攻击。我们观察到，敌意攻击和后门攻击的根本原因都存在于DNN模型的编码过程中，其中人类理解可以忽略的细微文本特征被较不健壮或特洛伊木马的模型错误地赋予了显著的权重。在此基础上，我们提出了一种统一的、自适应的防御框架，该框架能够有效地抵抗对手攻击和后门攻击。我们的方法利用重构模块来解决文本输入中潜在的恶意特征，同时保持原始语义的完整性。广泛的实验表明，我们的框架在各种恶意文本特征上的表现优于现有的面向样本的防御基线。



## **36. TrojanTime: Backdoor Attacks on Time Series Classification**

TrojanTime：对时间序列分类的后门攻击 cs.CR

13 pages, 3 figures, 3 tables

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00646v1) [paper-pdf](http://arxiv.org/pdf/2502.00646v1)

**Authors**: Chang Dong, Zechao Sun, Guangdong Bai, Shuying Piao, Weitong Chen, Wei Emma Zhang

**Abstract**: Time Series Classification (TSC) is highly vulnerable to backdoor attacks, posing significant security threats. Existing methods primarily focus on data poisoning during the training phase, designing sophisticated triggers to improve stealthiness and attack success rate (ASR). However, in practical scenarios, attackers often face restrictions in accessing training data. Moreover, it is a challenge for the model to maintain generalization ability on clean test data while remaining vulnerable to poisoned inputs when data is inaccessible. To address these challenges, we propose TrojanTime, a novel two-step training algorithm. In the first stage, we generate a pseudo-dataset using an external arbitrary dataset through target adversarial attacks. The clean model is then continually trained on this pseudo-dataset and its poisoned version. To ensure generalization ability, the second stage employs a carefully designed training strategy, combining logits alignment and batch norm freezing. We evaluate TrojanTime using five types of triggers across four TSC architectures in UCR benchmark datasets from diverse domains. The results demonstrate the effectiveness of TrojanTime in executing backdoor attacks while maintaining clean accuracy. Finally, to mitigate this threat, we propose a defensive unlearning strategy that effectively reduces the ASR while preserving clean accuracy.

摘要: 时间序列分类(TSC)非常容易受到后门攻击，构成严重的安全威胁。现有的方法主要集中在训练阶段的数据中毒，设计复杂的触发器来提高隐蔽性和攻击成功率(ASR)。然而，在实际场景中，攻击者在访问训练数据时经常面临限制。此外，当数据不可访问时，该模型在保持对干净测试数据的泛化能力的同时，仍然容易受到有毒输入的影响，这是一个挑战。为了应对这些挑战，我们提出了一种新的两步训练算法TrojanTime。在第一阶段，我们通过目标对抗性攻击，使用外部任意数据集生成伪数据集。然后，对这个伪数据集及其有毒版本持续训练CLEAN模型。为了确保泛化能力，第二阶段采用了精心设计的训练策略，将LOGITS对齐和批量范数冻结相结合。我们在来自不同领域的UCR基准数据集中，使用四种TSC体系结构的五种类型的触发器来评估TrojanTime。结果表明，木马时间在执行后门攻击的同时保持干净的准确性是有效的。最后，为了缓解这一威胁，我们提出了一种防御性遗忘策略，该策略有效地降低了ASR，同时保持了干净的准确性。



## **37. Actor Critic with Experience Replay-based automatic treatment planning for prostate cancer intensity modulated radiotherapy**

演员评论家与经验基于回放的前列腺癌调强放疗自动治疗计划 cs.LG

27 Pages, 8 Figures, 4 Tables

**SubmitDate**: 2025-02-01    [abs](http://arxiv.org/abs/2502.00346v1) [paper-pdf](http://arxiv.org/pdf/2502.00346v1)

**Authors**: Md Mainul Abrar, Parvat Sapkota, Damon Sprouts, Xun Jia, Yujie Chi

**Abstract**: Background: Real-time treatment planning in IMRT is challenging due to complex beam interactions. AI has improved automation, but existing models require large, high-quality datasets and lack universal applicability. Deep reinforcement learning (DRL) offers a promising alternative by mimicking human trial-and-error planning.   Purpose: Develop a stochastic policy-based DRL agent for automatic treatment planning with efficient training, broad applicability, and robustness against adversarial attacks using Fast Gradient Sign Method (FGSM).   Methods: Using the Actor-Critic with Experience Replay (ACER) architecture, the agent tunes treatment planning parameters (TPPs) in inverse planning. Training is based on prostate cancer IMRT cases, using dose-volume histograms (DVHs) as input. The model is trained on a single patient case, validated on two independent cases, and tested on 300+ plans across three datasets. Plan quality is assessed using ProKnow scores, and robustness is tested against adversarial attacks.   Results: Despite training on a single case, the model generalizes well. Before ACER-based planning, the mean plan score was 6.20$\pm$1.84; after, 93.09% of cases achieved a perfect score of 9, with a mean of 8.93$\pm$0.27. The agent effectively prioritizes optimal TPP tuning and remains robust against adversarial attacks.   Conclusions: The ACER-based DRL agent enables efficient, high-quality treatment planning in prostate cancer IMRT, demonstrating strong generalizability and robustness.

摘要: 背景：由于复杂的射束相互作用，调强放疗中的实时治疗计划是具有挑战性的。人工智能提高了自动化程度，但现有模型需要大量高质量的数据集，缺乏普遍适用性。深度强化学习(DRL)通过模仿人类的试错规划提供了一种很有前途的替代方法。目的：开发一种基于随机策略的DRL代理，用于基于快速梯度符号法(FGSM)的自动治疗规划，具有训练效率高、适用性广、抗攻击能力强等特点。方法：使用Acer-Critic with Experience Replay(ACER)架构，代理在反向计划中调整治疗计划参数(TPP)。培训以前列腺癌调强放疗病例为基础，使用剂量-体积直方图(DVH)作为输入。该模型在单个患者案例上进行了训练，在两个独立的案例上进行了验证，并在三个数据集的300多个计划上进行了测试。使用ProKnow分数评估计划质量，并测试对对手攻击的健壮性。结果：尽管对单个病例进行了训练，但该模型具有很好的泛化能力。在宏基计划之前，计划的平均得分为6.20$\pm$1.84；在宏基计划之后，93.09%的病例达到满分9分，平均得分为8.93$\pm$0.27。该代理有效地确定了最佳TPP调整的优先顺序，并保持了对对手攻击的健壮性。结论：基于宏碁的DRL试剂能够在前列腺癌IMRT中实现高效、高质量的治疗计划，显示出强大的通用性和稳健性。



## **38. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**

谜语我！检索增强一代的隐形会员推断 cs.CR

**SubmitDate**: 2025-02-01    [abs](http://arxiv.org/abs/2502.00306v1) [paper-pdf](http://arxiv.org/pdf/2502.00306v1)

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference.

摘要: 检索-增强生成(RAG)使大型语言模型(LLM)能够通过利用外部知识数据库生成接地响应，而无需更改模型参数。尽管没有权重调整防止通过模型参数泄漏，但它引入了推理对手在模型上下文中利用检索到的文档的风险。现有的成员关系推断和数据提取方法通常依赖于越狱或精心设计的非自然查询，这些查询可以很容易地被RAG系统中常见的查询重写技术检测到或阻止。在这项工作中，我们提出了询问攻击(IA)，这是一种针对RAG数据存储中的文档的成员关系推理技术。通过精心设计只能根据目标文档的存在来回答的自然文本查询，我们的方法只需30个查询即可成功推理，同时保持隐蔽性；直接的检测器识别来自现有方法的敌意提示的频率比我们的攻击生成的提示高约76倍。我们观察到，在不同的RAG配置中，TPR@1%的FPR比以前的推理攻击提高了2倍，而每个文档推理的成本都不到0.02美元。



## **39. Patch Synthesis for Property Repair of Deep Neural Networks**

用于深度神经网络属性修复的补丁合成 cs.LG

**SubmitDate**: 2025-02-01    [abs](http://arxiv.org/abs/2404.01642v2) [paper-pdf](http://arxiv.org/pdf/2404.01642v2)

**Authors**: Zhiming Chi, Jianan Ma, Pengfei Yang, Cheng-Chao Huang, Renjue Li, Xiaowei Huang, Lijun Zhang

**Abstract**: Deep neural networks (DNNs) are prone to various dependability issues, such as adversarial attacks, which hinder their adoption in safety-critical domains. Recently, NN repair techniques have been proposed to address these issues while preserving original performance by locating and modifying guilty neurons and their parameters. However, existing repair approaches are often limited to specific data sets and do not provide theoretical guarantees for the effectiveness of the repairs. To address these limitations, we introduce PatchPro, a novel patch-based approach for property-level repair of DNNs, focusing on local robustness. The key idea behind PatchPro is to construct patch modules that, when integrated with the original network, provide specialized repairs for all samples within the robustness neighborhood while maintaining the network's original performance. Our method incorporates formal verification and a heuristic mechanism for allocating patch modules, enabling it to defend against adversarial attacks and generalize to other inputs. PatchPro demonstrates superior efficiency, scalability, and repair success rates compared to existing DNN repair methods, i.e., realizing provable property-level repair for 100% cases across multiple high-dimensional datasets.

摘要: 深度神经网络(DNN)容易出现各种可靠性问题，如对抗性攻击，这阻碍了其在安全关键领域的应用。最近，神经网络修复技术被提出，通过定位和修改有罪神经元及其参数来解决这些问题，同时保持原始的性能。然而，现有的修复方法往往局限于特定的数据集，不能为修复的有效性提供理论上的保证。为了解决这些局限性，我们引入了PatchPro，这是一种新的基于补丁的DNN属性级修复方法，重点关注局部健壮性。PatchPro背后的关键思想是构建补丁模块，当与原始网络集成时，在保持网络原始性能的同时为健壮性邻域内的所有样本提供专门的修复。我们的方法结合了形式化的验证和分配补丁模块的启发式机制，使其能够防御对手攻击并推广到其他输入。与现有的DNN修复方法相比，PatchPro表现出更高的效率、可扩展性和修复成功率，即跨多个高维数据集实现100%可证明的属性级修复。



## **40. Robustifying ML-powered Network Classifiers with PANTS**

使用PANT来验证ML驱动的网络分类器 cs.CR

**SubmitDate**: 2025-02-01    [abs](http://arxiv.org/abs/2409.04691v3) [paper-pdf](http://arxiv.org/pdf/2409.04691v3)

**Authors**: Minhao Jin, Maria Apostolaki

**Abstract**: Multiple network management tasks, from resource allocation to intrusion detection, rely on some form of ML-based network traffic classification (MNC). Despite their potential, MNCs are vulnerable to adversarial inputs, which can lead to outages, poor decision-making, and security violations, among other issues. The goal of this paper is to help network operators assess and enhance the robustness of their MNC against adversarial inputs. The most critical step for this is generating inputs that can fool the MNC while being realizable under various threat models. Compared to other ML models, finding adversarial inputs against MNCs is more challenging due to the existence of non-differentiable components e.g., traffic engineering and the need to constrain inputs to preserve semantics and ensure reliability. These factors prevent the direct use of well-established gradient-based methods developed in adversarial ML (AML). To address these challenges, we introduce PANTS, a practical white-box framework that uniquely integrates AML techniques with Satisfiability Modulo Theories (SMT) solvers to generate adversarial inputs for MNCs. We also embed PANTS into an iterative adversarial training process that enhances the robustness of MNCs against adversarial inputs. PANTS is 70% and 2x more likely in median to find adversarial inputs against target MNCs compared to state-of-the-art baselines, namely Amoeba and BAP. PANTS improves the robustness of the target MNCs by 52.7% (even against attackers outside of what is considered during robustification) without sacrificing their accuracy.

摘要: 从资源分配到入侵检测的多个网络管理任务依赖于某种形式的基于ML的网络流量分类(MNC)。尽管跨国公司有潜力，但它们很容易受到敌意输入的影响，这可能会导致停电、糟糕的决策和违反安全规定等问题。本文的目的是帮助网络运营商评估和提高他们的MNC对敌意输入的健壮性。要做到这一点，最关键的一步是生成可以愚弄跨国公司的输入，同时在各种威胁模型下实现。与其他ML模型相比，发现针对跨国公司的敌意输入更具挑战性，这是因为存在不可区分的组件，例如流量工程，以及需要约束输入以保持语义和确保可靠性。这些因素阻碍了在对抗性ML(AML)中发展的基于梯度的方法的直接使用。为了应对这些挑战，我们引入了PANS，这是一个实用的白盒框架，它独特地将AML技术与可满足性模理论(SMT)求解器相结合，为跨国公司生成对抗性输入。我们还将裤子嵌入到一个迭代的对抗性训练过程中，以增强跨国公司对对抗性输入的健壮性。与最先进的基线，即变形虫和BAP相比，PANS在中位数中找到针对目标跨国公司的敌意输入的可能性分别高出70%和2倍。Pants在不牺牲精确度的情况下，将目标MNC的健壮性提高52.7%(即使是在攻击过程中考虑之外的攻击者)。



## **41. Redefining Machine Unlearning: A Conformal Prediction-Motivated Approach**

重新定义机器去学习：一种共形预测激励方法 cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19403v1) [paper-pdf](http://arxiv.org/pdf/2501.19403v1)

**Authors**: Yingdan Shi, Ren Wang

**Abstract**: Machine unlearning seeks to systematically remove specified data from a trained model, effectively achieving a state as though the data had never been encountered during training. While metrics such as Unlearning Accuracy (UA) and Membership Inference Attack (MIA) provide a baseline for assessing unlearning performance, they fall short of evaluating the completeness and reliability of forgetting. This is because the ground truth labels remain potential candidates within the scope of uncertainty quantification, leaving gaps in the evaluation of true forgetting. In this paper, we identify critical limitations in existing unlearning metrics and propose enhanced evaluation metrics inspired by conformal prediction. Our metrics can effectively capture the extent to which ground truth labels are excluded from the prediction set. Furthermore, we observe that many existing machine unlearning methods do not achieve satisfactory forgetting performance when evaluated with our new metrics. To address this, we propose an unlearning framework that integrates conformal prediction insights into Carlini & Wagner adversarial attack loss. Extensive experiments on the image classification task demonstrate that our enhanced metrics offer deeper insights into unlearning effectiveness, and that our unlearning framework significantly improves the forgetting quality of unlearning methods.

摘要: 机器遗忘寻求系统地从训练的模型中删除指定的数据，有效地达到一种状态，就好像在训练期间从未遇到过数据一样。遗忘准确率(UA)和成员关系推理攻击(MIA)等指标为评估遗忘绩效提供了一个基线，但它们并不能评估遗忘的完备性和可靠性。这是因为地面真相标签仍然是不确定性量化范围内的潜在候选者，在对真实遗忘的评估中留下了空白。在本文中，我们找出了现有遗忘度量的严重局限性，并提出了受保角预测启发的增强评估度量。我们的度量可以有效地捕捉到地面事实标签被排除在预测集中的程度。此外，我们观察到许多现有的机器遗忘方法在使用我们的新度量进行评估时并不能达到令人满意的遗忘性能。为了解决这一问题，我们提出了一个遗忘框架，该框架将共形预测的洞察力整合到Carlini&Wagner对手攻击的损失中。在图像分类任务上的广泛实验表明，我们增强的度量提供了对遗忘有效性的更深层次的洞察，并且我们的遗忘框架显著地改善了遗忘方法的遗忘质量。



## **42. SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders**

SAeUron：使用稀疏自动编码器的扩散模型中的可解释概念消除 cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.18052v2) [paper-pdf](http://arxiv.org/pdf/2501.18052v2)

**Authors**: Bartosz Cywiński, Kamil Deja

**Abstract**: Diffusion models, while powerful, can inadvertently generate harmful or undesirable content, raising significant ethical and safety concerns. Recent machine unlearning approaches offer potential solutions but often lack transparency, making it difficult to understand the changes they introduce to the base model. In this work, we introduce SAeUron, a novel method leveraging features learned by sparse autoencoders (SAEs) to remove unwanted concepts in text-to-image diffusion models. First, we demonstrate that SAEs, trained in an unsupervised manner on activations from multiple denoising timesteps of the diffusion model, capture sparse and interpretable features corresponding to specific concepts. Building on this, we propose a feature selection method that enables precise interventions on model activations to block targeted content while preserving overall performance. Evaluation with the competitive UnlearnCanvas benchmark on object and style unlearning highlights SAeUron's state-of-the-art performance. Moreover, we show that with a single SAE, we can remove multiple concepts simultaneously and that in contrast to other methods, SAeUron mitigates the possibility of generating unwanted content, even under adversarial attack. Code and checkpoints are available at: https://github.com/cywinski/SAeUron.

摘要: 传播模式虽然强大，但可能会在不经意间产生有害或不良内容，引发重大的道德和安全问题。最近的机器遗忘方法提供了潜在的解决方案，但往往缺乏透明度，使得很难理解它们给基本模型带来的变化。在这项工作中，我们引入了SAeUron，这是一种新的方法，利用稀疏自动编码器(SAE)学习的特征来消除文本到图像扩散模型中不需要的概念。首先，我们证明了在非监督方式下对扩散模型的多个去噪时间步长的激活进行训练的SAE能够捕获与特定概念相对应的稀疏且可解释的特征。在此基础上，我们提出了一种特征选择方法，该方法能够对模型激活进行精确干预，以阻止目标内容，同时保持整体性能。使用具有竞争力的UnlearnCanvas基准对对象和风格遗忘进行评估，突出了SAeUron最先进的性能。此外，我们证明了使用单个SAE，我们可以同时删除多个概念，并且与其他方法相比，SAeUron降低了生成不想要的内容的可能性，即使在敌意攻击下也是如此。代码和检查点可在以下网址获得：https://github.com/cywinski/SAeUron.



## **43. UniGuard: Towards Universal Safety Guardrails for Jailbreak Attacks on Multimodal Large Language Models**

UniGuard：为多模式大型语言模型越狱攻击建立通用安全护栏 cs.CL

14 pages

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2411.01703v2) [paper-pdf](http://arxiv.org/pdf/2411.01703v2)

**Authors**: Sejoon Oh, Yiqiao Jin, Megha Sharma, Donghyun Kim, Eric Ma, Gaurav Verma, Srijan Kumar

**Abstract**: Multimodal large language models (MLLMs) have revolutionized vision-language understanding but remain vulnerable to multimodal jailbreak attacks, where adversarial inputs are meticulously crafted to elicit harmful or inappropriate responses. We propose UniGuard, a novel multimodal safety guardrail that jointly considers the unimodal and cross-modal harmful signals. UniGuard trains a multimodal guardrail to minimize the likelihood of generating harmful responses in a toxic corpus. The guardrail can be seamlessly applied to any input prompt during inference with minimal computational costs. Extensive experiments demonstrate the generalizability of UniGuard across multiple modalities, attack strategies, and multiple state-of-the-art MLLMs, including LLaVA, Gemini Pro, GPT-4o, MiniGPT-4, and InstructBLIP. Notably, this robust defense mechanism maintains the models' overall vision-language understanding capabilities.

摘要: 多模式大型语言模型（MLLM）彻底改变了视觉语言理解，但仍然容易受到多模式越狱攻击的影响，其中对抗性输入经过精心设计，以引发有害或不当的反应。我们提出了UniGuard，这是一种新型的多模式安全护栏，它联合考虑了单模式和跨模式有害信号。UniGuard训练多模式护栏，以最大限度地降低有毒主体中产生有害反应的可能性。护栏可以无缝地应用于推理期间的任何输入提示，并且计算成本最低。大量实验证明了UniGuard在多种模式、攻击策略和多种最先进的MLLM中的通用性，包括LLaVA、Gemini Pro、GPT-4 o、MiniGPT-4和DirecectBLIP。值得注意的是，这种强大的防御机制维持了模型的整体视觉语言理解能力。



## **44. An Empirical Game-Theoretic Analysis of Autonomous Cyber-Defence Agents**

自主网络防御代理的经验游戏理论分析 cs.AI

21 pages, 17 figures, 10 tables

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19206v1) [paper-pdf](http://arxiv.org/pdf/2501.19206v1)

**Authors**: Gregory Palmer, Luke Swaby, Daniel J. B. Harrold, Matthew Stewart, Alex Hiles, Chris Willis, Ian Miles, Sara Farmer

**Abstract**: The recent rise in increasingly sophisticated cyber-attacks raises the need for robust and resilient autonomous cyber-defence (ACD) agents. Given the variety of cyber-attack tactics, techniques and procedures (TTPs) employed, learning approaches that can return generalisable policies are desirable. Meanwhile, the assurance of ACD agents remains an open challenge. We address both challenges via an empirical game-theoretic analysis of deep reinforcement learning (DRL) approaches for ACD using the principled double oracle (DO) algorithm. This algorithm relies on adversaries iteratively learning (approximate) best responses against each others' policies; a computationally expensive endeavour for autonomous cyber operations agents. In this work we introduce and evaluate a theoretically-sound, potential-based reward shaping approach to expedite this process. In addition, given the increasing number of open-source ACD-DRL approaches, we extend the DO formulation to allow for multiple response oracles (MRO), providing a framework for a holistic evaluation of ACD approaches.

摘要: 最近日益复杂的网络攻击的增加增加了对强大和有弹性的自主网络防御(ACD)代理的需求。考虑到所使用的网络攻击策略、技术和程序(TTP)的多样性，可以返回通用策略的学习方法是可取的。与此同时，ACD代理商的保证仍然是一个开放的挑战。我们通过使用原则性双先知(DO)算法对ACD的深度强化学习(DRL)方法进行经验博弈论分析来解决这两个挑战。该算法依赖于对手根据彼此的策略迭代地学习(近似)最佳响应；对于自主网络操作代理来说，这是一种计算代价高昂的努力。在这项工作中，我们介绍和评估了一种理论上合理的、基于潜力的报酬形成方法，以加快这一过程。此外，鉴于开源ACD-DRL方法的数量不断增加，我们扩展了DO公式以允许多个响应先知(MRO)，为ACD方法的整体评估提供了一个框架。



## **45. Enhancing Model Defense Against Jailbreaks with Proactive Safety Reasoning**

利用主动安全推理增强模型对越狱的防御 cs.CR

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19180v1) [paper-pdf](http://arxiv.org/pdf/2501.19180v1)

**Authors**: Xianglin Yang, Gelei Deng, Jieming Shi, Tianwei Zhang, Jin Song Dong

**Abstract**: Large language models (LLMs) are vital for a wide range of applications yet remain susceptible to jailbreak threats, which could lead to the generation of inappropriate responses. Conventional defenses, such as refusal and adversarial training, often fail to cover corner cases or rare domains, leaving LLMs still vulnerable to more sophisticated attacks. We propose a novel defense strategy, Safety Chain-of-Thought (SCoT), which harnesses the enhanced \textit{reasoning capabilities} of LLMs for proactive assessment of harmful inputs, rather than simply blocking them. SCoT augments any refusal training datasets to critically analyze the intent behind each request before generating answers. By employing proactive reasoning, SCoT enhances the generalization of LLMs across varied harmful queries and scenarios not covered in the safety alignment corpus. Additionally, it generates detailed refusals specifying the rules violated. Comparative evaluations show that SCoT significantly surpasses existing defenses, reducing vulnerability to out-of-distribution issues and adversarial manipulations while maintaining strong general capabilities.

摘要: 大型语言模型(LLM)对于广泛的应用至关重要，但仍然容易受到越狱威胁的影响，这可能会导致产生不适当的响应。常规防御，如拒绝和对抗性训练，往往无法覆盖角落案例或稀有领域，使LLM仍然容易受到更复杂的攻击。我们提出了一种新的防御策略，安全思想链(SCOT)，它利用LLMS增强的\文本{推理能力}来主动评估有害输入，而不是简单地阻止它们。SCOT增加了任何拒绝训练数据集，以便在生成答案之前批判性地分析每个请求背后的意图。通过使用主动推理，SCOT增强了LLMS在安全匹配语料库中未涵盖的各种有害查询和场景中的泛化。此外，它还生成详细的拒绝，指定违反的规则。比较评估表明，SCOT大大超过了现有的防御系统，在保持强大的一般能力的同时，减少了对分配外问题和对抗性操纵的脆弱性。



## **46. Imitation Game for Adversarial Disillusion with Multimodal Generative Chain-of-Thought Role-Play**

具有多模式生成思想链角色扮演的对抗幻灭模仿游戏 cs.AI

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19143v1) [paper-pdf](http://arxiv.org/pdf/2501.19143v1)

**Authors**: Ching-Chun Chang, Fan-Yun Chen, Shih-Hong Gu, Kai Gao, Hanrui Wang, Isao Echizen

**Abstract**: As the cornerstone of artificial intelligence, machine perception confronts a fundamental threat posed by adversarial illusions. These adversarial attacks manifest in two primary forms: deductive illusion, where specific stimuli are crafted based on the victim model's general decision logic, and inductive illusion, where the victim model's general decision logic is shaped by specific stimuli. The former exploits the model's decision boundaries to create a stimulus that, when applied, interferes with its decision-making process. The latter reinforces a conditioned reflex in the model, embedding a backdoor during its learning phase that, when triggered by a stimulus, causes aberrant behaviours. The multifaceted nature of adversarial illusions calls for a unified defence framework, addressing vulnerabilities across various forms of attack. In this study, we propose a disillusion paradigm based on the concept of an imitation game. At the heart of the imitation game lies a multimodal generative agent, steered by chain-of-thought reasoning, which observes, internalises and reconstructs the semantic essence of a sample, liberated from the classic pursuit of reversing the sample to its original state. As a proof of concept, we conduct experimental simulations using a multimodal generative dialogue agent and evaluates the methodology under a variety of attack scenarios.

摘要: 作为人工智能的基石，机器感知面临着对抗性错觉带来的根本性威胁。这些对抗性攻击主要表现为两种形式：演绎错觉和归纳错觉，演绎错觉是根据受害者模型的一般决策逻辑制定特定刺激的，归纳错觉是受害者模型的一般决策逻辑由特定刺激塑造的。前者利用模型的决策边界来创造一种刺激，当应用时，会干扰其决策过程。后者强化了模型中的条件反射，在其学习阶段嵌入了一个后门，当被刺激触发时，后门会导致异常行为。对抗性幻觉的多面性要求建立统一的防御框架，处理各种攻击形式的脆弱性。在本研究中，我们提出了一个基于模仿游戏概念的幻灭范式。模仿游戏的核心是一个由思维链推理控制的多模式生成主体，它观察、内化和重建样本的语义本质，从将样本反转到原始状态的经典追求中解放出来。作为概念验证，我们使用多模式生成性对话代理进行了实验模拟，并在各种攻击场景下对该方法进行了评估。



## **47. Average Certified Radius is a Poor Metric for Randomized Smoothing**

平均认证半径对于随机平滑来说是一个较差的指标 cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2410.06895v2) [paper-pdf](http://arxiv.org/pdf/2410.06895v2)

**Authors**: Chenhao Sun, Yuhao Mao, Mark Niklas Müller, Martin Vechev

**Abstract**: Randomized smoothing is a popular approach for providing certified robustness guarantees against adversarial attacks, and has become an active area of research. Over the past years, the average certified radius (ACR) has emerged as the most important metric for comparing methods and tracking progress in the field. However, in this work, for the first time we show that ACR is a poor metric for evaluating robustness guarantees provided by randomized smoothing. We theoretically prove not only that a trivial classifier can have arbitrarily large ACR, but also that ACR is much more sensitive to improvements on easy samples than on hard ones. Empirically, we confirm that existing training strategies, though improving ACR with different approaches, reduce the model's robustness on hard samples consistently. To strengthen our conclusion, we propose strategies, including explicitly discarding hard samples, reweighting the dataset with approximate certified radius, and extreme optimization for easy samples, to achieve state-of-the-art ACR, without training for robustness on the full data distribution. Overall, our results suggest that ACR has introduced a strong undesired bias to the field, and its application should be discontinued when evaluating randomized smoothing.

摘要: 随机化平滑是一种流行的方法，用于提供对敌意攻击的健壮性保证，并已成为一个活跃的研究领域。在过去的几年里，平均认证半径(ACR)已经成为比较方法和跟踪该领域进展的最重要的衡量标准。然而，在这项工作中，我们第一次表明ACR不是评估随机平滑所提供的稳健性保证的一个很差的度量。我们不仅从理论上证明了一个平凡的分类器可以有任意大的ACR，而且ACR对简单样本的改进比对困难样本的改进更敏感。经验证明，现有的训练策略虽然用不同的方法提高了ACR，但一致地降低了模型在硬样本上的稳健性。为了加强我们的结论，我们提出了一些策略，包括显式丢弃硬样本，用近似认证半径重新加权数据集，以及对简单样本进行极端优化，以实现最先进的ACR，而不需要对完全数据分布的稳健性进行培训。总体而言，我们的结果表明，ACR给该领域带来了强烈的不良偏差，在评估随机平滑时，应该停止它的应用。



## **48. Understanding Oversmoothing in GNNs as Consensus in Opinion Dynamics**

将GNN中的过度平滑理解为观点动态中的共识 cs.LG

23 pages, 3 figures

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19089v1) [paper-pdf](http://arxiv.org/pdf/2501.19089v1)

**Authors**: Keqin Wang, Yulong Yang, Ishan Saha, Christine Allen-Blanchette

**Abstract**: In contrast to classes of neural networks where the learned representations become increasingly expressive with network depth, the learned representations in graph neural networks (GNNs), tend to become increasingly similar. This phenomena, known as oversmoothing, is characterized by learned representations that cannot be reliably differentiated leading to reduced predictive performance. In this paper, we propose an analogy between oversmoothing in GNNs and consensus or agreement in opinion dynamics. Through this analogy, we show that the message passing structure of recent continuous-depth GNNs is equivalent to a special case of opinion dynamics (i.e., linear consensus models) which has been theoretically proven to converge to consensus (i.e., oversmoothing) for all inputs. Using the understanding developed through this analogy, we design a new continuous-depth GNN model based on nonlinear opinion dynamics and prove that our model, which we call behavior-inspired message passing neural network (BIMP) circumvents oversmoothing for general inputs. Through extensive experiments, we show that BIMP is robust to oversmoothing and adversarial attack, and consistently outperforms competitive baselines on numerous benchmarks.

摘要: 与神经网络类中学习的表示随着网络深度的增加而变得越来越有表现力相比，图神经网络(GNN)中的学习表示往往变得越来越相似。这种现象被称为过平滑，其特征是学习的表示不能可靠地区分，从而导致预测性能降低。在这篇文章中，我们提出了GNN中的超平滑与意见动力学中的共识或一致之间的类比。通过这个类比，我们证明了最近的连续深度GNN的消息传递结构等价于意见动力学(即线性共识模型)的一个特例，该模型已被理论证明对于所有输入都收敛到共识(即过度平滑)。利用这种类比得到的理解，我们设计了一种新的基于非线性观点动力学的连续深度GNN模型，并证明了我们的模型，我们称之为行为启发消息传递神经网络(BIMP)，避免了对一般输入的过度平滑。通过大量的实验，我们证明了BIMP对过平滑和敌意攻击具有较强的鲁棒性，并且在许多基准测试中始终优于竞争基线。



## **49. Concept Steerers: Leveraging K-Sparse Autoencoders for Controllable Generations**

概念掌舵者：利用K稀疏自动编码器实现可控发电 cs.CV

15 pages, 16 figures

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19066v1) [paper-pdf](http://arxiv.org/pdf/2501.19066v1)

**Authors**: Dahye Kim, Deepti Ghadiyaram

**Abstract**: Despite the remarkable progress in text-to-image generative models, they are prone to adversarial attacks and inadvertently generate unsafe, unethical content. Existing approaches often rely on fine-tuning models to remove specific concepts, which is computationally expensive, lack scalability, and/or compromise generation quality. In this work, we propose a novel framework leveraging k-sparse autoencoders (k-SAEs) to enable efficient and interpretable concept manipulation in diffusion models. Specifically, we first identify interpretable monosemantic concepts in the latent space of text embeddings and leverage them to precisely steer the generation away or towards a given concept (e.g., nudity) or to introduce a new concept (e.g., photographic style). Through extensive experiments, we demonstrate that our approach is very simple, requires no retraining of the base model nor LoRA adapters, does not compromise the generation quality, and is robust to adversarial prompt manipulations. Our method yields an improvement of $\mathbf{20.01\%}$ in unsafe concept removal, is effective in style manipulation, and is $\mathbf{\sim5}$x faster than current state-of-the-art.

摘要: 尽管文本到图像的生成模型取得了显著的进步，但它们容易受到敌意攻击，并在不经意间生成不安全、不道德的内容。现有的方法通常依赖于微调模型来删除特定的概念，这是计算昂贵、缺乏可伸缩性和/或损害生成质量的。在这项工作中，我们提出了一个新的框架，利用k-稀疏自动编码器(k-SAE)来实现扩散模型中高效和可解释的概念操作。具体地说，我们首先在文本嵌入的潜在空间中识别可解释的单一语义概念，并利用它们来精确地引导这一代远离或转向给定的概念(例如，裸体)或引入新的概念(例如，摄影风格)。通过大量的实验，我们证明了我们的方法非常简单，不需要重新训练基本模型和LORA适配器，不会影响生成质量，并且对敌意提示操作具有很强的鲁棒性。我们的方法在去除不安全概念方面得到了改进，在样式处理方面是有效的，并且比目前最先进的水平快了$\mathbf{2 0.0 1}$。



## **50. Towards the Worst-case Robustness of Large Language Models**

走向大型语言模型的最坏情况稳健性 cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19040v1) [paper-pdf](http://arxiv.org/pdf/2501.19040v1)

**Authors**: Huanran Chen, Yinpeng Dong, Zeming Wei, Hang Su, Jun Zhu

**Abstract**: Recent studies have revealed the vulnerability of Large Language Models (LLMs) to adversarial attacks, where the adversary crafts specific input sequences to induce harmful, violent, private, or incorrect outputs. Although various defenses have been proposed, they have not been evaluated by strong adaptive attacks, leaving the worst-case robustness of LLMs still intractable. By developing a stronger white-box attack, our evaluation results indicate that most typical defenses achieve nearly 0\% robustness.To solve this, we propose \textit{DiffTextPure}, a general defense that diffuses the (adversarial) input prompt using any pre-defined smoothing distribution, and purifies the diffused input using a pre-trained language model. Theoretically, we derive tight robustness lower bounds for all smoothing distributions using Fractal Knapsack or 0-1 Knapsack solvers. Under this framework, we certify the robustness of a specific case -- smoothing LLMs using a uniform kernel -- against \textit{any possible attack} with an average $\ell_0$ perturbation of 2.02 or an average suffix length of 6.41.

摘要: 最近的研究揭示了大型语言模型(LLM)在敌意攻击中的脆弱性，在这种攻击中，对手精心制作特定的输入序列来诱导有害的、暴力的、隐私的或错误的输出。虽然已经提出了各种防御措施，但它们还没有通过强自适应攻击进行评估，这使得LLM的最坏情况下的稳健性仍然很难解决。通过开发更强的白盒攻击，我们的评估结果表明，大多数典型的防御措施都达到了近0的稳健性，为了解决这个问题，我们提出了一种通用的防御措施，它使用任何预定义的平滑分布来扩散(对抗性的)输入提示，并使用预先训练的语言模型来净化扩散的输入。理论上，我们使用分形型背包或0-1背包求解器得到了所有光滑分布的紧鲁棒下界。在此框架下，我们证明了一种特殊情况--使用统一核的平滑LLMS--对平均$\ELL_0$扰动为2.02或平均后缀长度为6.41的文本{任何可能的攻击}的健壮性。



