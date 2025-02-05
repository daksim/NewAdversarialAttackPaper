# Latest Adversarial Attack Papers
**update at 2025-02-05 12:10:49**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. OVERTHINKING: Slowdown Attacks on Reasoning LLMs**

cs.LG

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02542v1) [paper-pdf](http://arxiv.org/pdf/2502.02542v1)

**Authors**: Abhinav Kumar, Jaechul Roh, Ali Naseh, Marzena Karpinska, Mohit Iyyer, Amir Houmansadr, Eugene Bagdasarian

**Abstract**: We increase overhead for applications that rely on reasoning LLMs-we force models to spend an amplified number of reasoning tokens, i.e., "overthink", to respond to the user query while providing contextually correct answers. The adversary performs an OVERTHINK attack by injecting decoy reasoning problems into the public content that is used by the reasoning LLM (e.g., for RAG applications) during inference time. Due to the nature of our decoy problems (e.g., a Markov Decision Process), modified texts do not violate safety guardrails. We evaluated our attack across closed-(OpenAI o1, o1-mini, o3-mini) and open-(DeepSeek R1) weights reasoning models on the FreshQA and SQuAD datasets. Our results show up to 46x slowdown and high transferability of the attack across models. To protect applications, we discuss and implement defenses leveraging LLM-based and system design approaches. Finally, we discuss societal, financial, and energy impacts of OVERTHINK attack which could amplify the costs for third party applications operating reasoning models.



## **2. Uncertainty Quantification for Collaborative Object Detection Under Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02537v1) [paper-pdf](http://arxiv.org/pdf/2502.02537v1)

**Authors**: Huiqun Huang, Cong Chen, Jean-Philippe Monteuuis, Jonathan Petit, Fei Miao

**Abstract**: Collaborative Object Detection (COD) and collaborative perception can integrate data or features from various entities, and improve object detection accuracy compared with individual perception. However, adversarial attacks pose a potential threat to the deep learning COD models, and introduce high output uncertainty. With unknown attack models, it becomes even more challenging to improve COD resiliency and quantify the output uncertainty for highly dynamic perception scenes such as autonomous vehicles. In this study, we propose the Trusted Uncertainty Quantification in Collaborative Perception framework (TUQCP). TUQCP leverages both adversarial training and uncertainty quantification techniques to enhance the adversarial robustness of existing COD models. More specifically, TUQCP first adds perturbations to the shared information of randomly selected agents during object detection collaboration by adversarial training. TUQCP then alleviates the impacts of adversarial attacks by providing output uncertainty estimation through learning-based module and uncertainty calibration through conformal prediction. Our framework works for early and intermediate collaboration COD models and single-agent object detection models. We evaluate TUQCP on V2X-Sim, a comprehensive collaborative perception dataset for autonomous driving, and demonstrate a 80.41% improvement in object detection accuracy compared to the baselines under the same adversarial attacks. TUQCP demonstrates the importance of uncertainty quantification to COD under adversarial attacks.



## **3. The TIP of the Iceberg: Revealing a Hidden Class of Task-in-Prompt Adversarial Attacks on LLMs**

cs.CR

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2501.18626v3) [paper-pdf](http://arxiv.org/pdf/2501.18626v3)

**Authors**: Sergey Berezin, Reza Farahbakhsh, Noel Crespi

**Abstract**: We present a novel class of jailbreak adversarial attacks on LLMs, termed Task-in-Prompt (TIP) attacks. Our approach embeds sequence-to-sequence tasks (e.g., cipher decoding, riddles, code execution) into the model's prompt to indirectly generate prohibited inputs. To systematically assess the effectiveness of these attacks, we introduce the PHRYGE benchmark. We demonstrate that our techniques successfully circumvent safeguards in six state-of-the-art language models, including GPT-4o and LLaMA 3.2. Our findings highlight critical weaknesses in current LLM safety alignments and underscore the urgent need for more sophisticated defence strategies.   Warning: this paper contains examples of unethical inquiries used solely for research purposes.



## **4. Medical Multimodal Model Stealing Attacks via Adversarial Domain Alignment**

cs.CR

Accepted at AAAI 2025

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02438v1) [paper-pdf](http://arxiv.org/pdf/2502.02438v1)

**Authors**: Yaling Shen, Zhixiong Zhuang, Kun Yuan, Maria-Irina Nicolae, Nassir Navab, Nicolas Padoy, Mario Fritz

**Abstract**: Medical multimodal large language models (MLLMs) are becoming an instrumental part of healthcare systems, assisting medical personnel with decision making and results analysis. Models for radiology report generation are able to interpret medical imagery, thus reducing the workload of radiologists. As medical data is scarce and protected by privacy regulations, medical MLLMs represent valuable intellectual property. However, these assets are potentially vulnerable to model stealing, where attackers aim to replicate their functionality via black-box access. So far, model stealing for the medical domain has focused on classification; however, existing attacks are not effective against MLLMs. In this paper, we introduce Adversarial Domain Alignment (ADA-STEAL), the first stealing attack against medical MLLMs. ADA-STEAL relies on natural images, which are public and widely available, as opposed to their medical counterparts. We show that data augmentation with adversarial noise is sufficient to overcome the data distribution gap between natural images and the domain-specific distribution of the victim MLLM. Experiments on the IU X-RAY and MIMIC-CXR radiology datasets demonstrate that Adversarial Domain Alignment enables attackers to steal the medical MLLM without any access to medical data.



## **5. Rule-ATT&CK Mapper (RAM): Mapping SIEM Rules to TTPs Using LLMs**

cs.CR

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02337v1) [paper-pdf](http://arxiv.org/pdf/2502.02337v1)

**Authors**: Prasanna N. Wudali, Moshe Kravchik, Ehud Malul, Parth A. Gandhi, Yuval Elovici, Asaf Shabtai

**Abstract**: The growing frequency of cyberattacks has heightened the demand for accurate and efficient threat detection systems. SIEM platforms are important for analyzing log data and detecting adversarial activities through rule-based queries, also known as SIEM rules. The efficiency of the threat analysis process relies heavily on mapping these SIEM rules to the relevant attack techniques in the MITRE ATT&CK framework. Inaccurate annotation of SIEM rules can result in the misinterpretation of attacks, increasing the likelihood that threats will be overlooked. Existing solutions for annotating SIEM rules with MITRE ATT&CK technique labels have notable limitations: manual annotation of SIEM rules is both time-consuming and prone to errors, and ML-based approaches mainly focus on annotating unstructured free text sources rather than structured data like SIEM rules. Structured data often contains limited information, further complicating the annotation process and making it a challenging task. To address these challenges, we propose Rule-ATT&CK Mapper (RAM), a novel framework that leverages LLMs to automate the mapping of structured SIEM rules to MITRE ATT&CK techniques. RAM's multi-stage pipeline, which was inspired by the prompt chaining technique, enhances mapping accuracy without requiring LLM pre-training or fine-tuning. Using the Splunk Security Content dataset, we evaluate RAM's performance using several LLMs, including GPT-4-Turbo, Qwen, IBM Granite, and Mistral. Our evaluation highlights GPT-4-Turbo's superior performance, which derives from its enriched knowledge base, and an ablation study emphasizes the importance of external contextual knowledge in overcoming the limitations of LLMs' implicit knowledge for domain-specific tasks. These findings demonstrate RAM's potential in automating cybersecurity workflows and provide valuable insights for future advancements in this field.



## **6. FRAUD-RLA: A new reinforcement learning adversarial attack against credit card fraud detection**

cs.LG

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02290v1) [paper-pdf](http://arxiv.org/pdf/2502.02290v1)

**Authors**: Daniele Lunghi, Yannick Molinghen, Alkis Simitsis, Tom Lenaerts, Gianluca Bontempi

**Abstract**: Adversarial attacks pose a significant threat to data-driven systems, and researchers have spent considerable resources studying them. Despite its economic relevance, this trend largely overlooked the issue of credit card fraud detection. To address this gap, we propose a new threat model that demonstrates the limitations of existing attacks and highlights the necessity to investigate new approaches. We then design a new adversarial attack for credit card fraud detection, employing reinforcement learning to bypass classifiers. This attack, called FRAUD-RLA, is designed to maximize the attacker's reward by optimizing the exploration-exploitation tradeoff and working with significantly less required knowledge than competitors. Our experiments, conducted on three different heterogeneous datasets and against two fraud detection systems, indicate that FRAUD-RLA is effective, even considering the severe limitations imposed by our threat model.



## **7. Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization**

cs.CV

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02096v1) [paper-pdf](http://arxiv.org/pdf/2502.02096v1)

**Authors**: Yixiao Chen, Shikun Sun, Jianshu Li, Ruoyu Li, Zhe Li, Junliang Xing

**Abstract**: Adversarial attacks are widely used to evaluate model robustness, and in black-box scenarios, the transferability of these attacks becomes crucial. Existing generator-based attacks have excellent generalization and transferability due to their instance-agnostic nature. However, when training generators for multi-target tasks, the success rate of transfer attacks is relatively low due to the limitations of the model's capacity. To address these challenges, we propose a novel Dual-Flow framework for multi-target instance-agnostic adversarial attacks, utilizing Cascading Distribution Shift Training to develop an adversarial velocity function. Extensive experiments demonstrate that Dual-Flow significantly improves transferability over previous multi-target generative attacks. For example, it increases the success rate from Inception-v3 to ResNet-152 by 34.58%. Furthermore, our attack method, such as adversarially trained models, shows substantially stronger robustness against defense mechanisms.



## **8. Model Supply Chain Poisoning: Backdooring Pre-trained Models via Embedding Indistinguishability**

cs.CR

ACM Web Conference 2025 (Oral)

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2401.15883v3) [paper-pdf](http://arxiv.org/pdf/2401.15883v3)

**Authors**: Hao Wang, Shangwei Guo, Jialing He, Hangcheng Liu, Tianwei Zhang, Tao Xiang

**Abstract**: Pre-trained models (PTMs) are widely adopted across various downstream tasks in the machine learning supply chain. Adopting untrustworthy PTMs introduces significant security risks, where adversaries can poison the model supply chain by embedding hidden malicious behaviors (backdoors) into PTMs. However, existing backdoor attacks to PTMs can only achieve partially task-agnostic and the embedded backdoors are easily erased during the fine-tuning process. This makes it challenging for the backdoors to persist and propagate through the supply chain. In this paper, we propose a novel and severer backdoor attack, TransTroj, which enables the backdoors embedded in PTMs to efficiently transfer in the model supply chain. In particular, we first formalize this attack as an indistinguishability problem between poisoned and clean samples in the embedding space. We decompose embedding indistinguishability into pre- and post-indistinguishability, representing the similarity of the poisoned and reference embeddings before and after the attack. Then, we propose a two-stage optimization that separately optimizes triggers and victim PTMs to achieve embedding indistinguishability. We evaluate TransTroj on four PTMs and six downstream tasks. Experimental results show that our method significantly outperforms SOTA task-agnostic backdoor attacks -- achieving nearly 100% attack success rate on most downstream tasks -- and demonstrates robustness under various system settings. Our findings underscore the urgent need to secure the model supply chain against such transferable backdoor attacks. The code is available at https://github.com/haowang-cqu/TransTroj .



## **9. Multi-Domain Graph Foundation Models: Robust Knowledge Transfer via Topology Alignment**

cs.SI

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02017v1) [paper-pdf](http://arxiv.org/pdf/2502.02017v1)

**Authors**: Shuo Wang, Bokui Wang, Zhixiang Shen, Boyan Deng, Zhao Kang

**Abstract**: Recent advances in CV and NLP have inspired researchers to develop general-purpose graph foundation models through pre-training across diverse domains. However, a fundamental challenge arises from the substantial differences in graph topologies across domains. Additionally, real-world graphs are often sparse and prone to noisy connections and adversarial attacks. To address these issues, we propose the Multi-Domain Graph Foundation Model (MDGFM), a unified framework that aligns and leverages cross-domain topological information to facilitate robust knowledge transfer. MDGFM bridges different domains by adaptively balancing features and topology while refining original graphs to eliminate noise and align topological structures. To further enhance knowledge transfer, we introduce an efficient prompt-tuning approach. By aligning topologies, MDGFM not only improves multi-domain pre-training but also enables robust knowledge transfer to unseen domains. Theoretical analyses provide guarantees of MDGFM's effectiveness and domain generalization capabilities. Extensive experiments on both homophilic and heterophilic graph datasets validate the robustness and efficacy of our method.



## **10. Evaluating the Robustness of the "Ensemble Everything Everywhere" Defense**

cs.LG

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2411.14834v2) [paper-pdf](http://arxiv.org/pdf/2411.14834v2)

**Authors**: Jie Zhang, Christian Schlarmann, Kristina Nikolić, Nicholas Carlini, Francesco Croce, Matthias Hein, Florian Tramèr

**Abstract**: Ensemble everything everywhere is a defense to adversarial examples that was recently proposed to make image classifiers robust. This defense works by ensembling a model's intermediate representations at multiple noisy image resolutions, producing a single robust classification. This defense was shown to be effective against multiple state-of-the-art attacks. Perhaps even more convincingly, it was shown that the model's gradients are perceptually aligned: attacks against the model produce noise that perceptually resembles the targeted class.   In this short note, we show that this defense is not robust to adversarial attack. We first show that the defense's randomness and ensembling method cause severe gradient masking. We then use standard adaptive attack techniques to reduce the defense's robust accuracy from 48% to 14% on CIFAR-100 and from 62% to 11% on CIFAR-10, under the $\ell_\infty$-norm threat model with $\varepsilon=8/255$.



## **11. Adversarial Reasoning at Jailbreaking Time**

cs.LG

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01633v1) [paper-pdf](http://arxiv.org/pdf/2502.01633v1)

**Authors**: Mahdi Sabbaghi, Paul Kassianik, George Pappas, Yaron Singer, Amin Karbasi, Hamed Hassani

**Abstract**: As large language models (LLMs) are becoming more capable and widespread, the study of their failure cases is becoming increasingly important. Recent advances in standardizing, measuring, and scaling test-time compute suggest new methodologies for optimizing models to achieve high performance on hard tasks. In this paper, we apply these advances to the task of model jailbreaking: eliciting harmful responses from aligned LLMs. We develop an adversarial reasoning approach to automatic jailbreaking via test-time computation that achieves SOTA attack success rates (ASR) against many aligned LLMs, even the ones that aim to trade inference-time compute for adversarial robustness. Our approach introduces a new paradigm in understanding LLM vulnerabilities, laying the foundation for the development of more robust and trustworthy AI systems.



## **12. Robust-LLaVA: On the Effectiveness of Large-Scale Robust Image Encoders for Multi-modal Large Language Models**

cs.CV

Under Review

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01576v1) [paper-pdf](http://arxiv.org/pdf/2502.01576v1)

**Authors**: Hashmat Shadab Malik, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar, Fahad Khan, Salman Khan

**Abstract**: Multi-modal Large Language Models (MLLMs) excel in vision-language tasks but remain vulnerable to visual adversarial perturbations that can induce hallucinations, manipulate responses, or bypass safety mechanisms. Existing methods seek to mitigate these risks by applying constrained adversarial fine-tuning to CLIP vision encoders on ImageNet-scale data, ensuring their generalization ability is preserved. However, this limited adversarial training restricts robustness and broader generalization. In this work, we explore an alternative approach of leveraging existing vision classification models that have been adversarially pre-trained on large-scale data. Our analysis reveals two principal contributions: (1) the extensive scale and diversity of adversarial pre-training enables these models to demonstrate superior robustness against diverse adversarial threats, ranging from imperceptible perturbations to advanced jailbreaking attempts, without requiring additional adversarial training, and (2) end-to-end MLLM integration with these robust models facilitates enhanced adaptation of language components to robust visual features, outperforming existing plug-and-play methodologies on complex reasoning tasks. Through systematic evaluation across visual question-answering, image captioning, and jail-break attacks, we demonstrate that MLLMs trained with these robust models achieve superior adversarial robustness while maintaining favorable clean performance. Our framework achieves 2x and 1.5x average robustness gains in captioning and VQA tasks, respectively, and delivers over 10% improvement against jailbreak attacks. Code and pretrained models will be available at https://github.com/HashmatShadab/Robust-LLaVA.



## **13. Quantum Quandaries: Unraveling Encoding Vulnerabilities in Quantum Neural Networks**

quant-ph

7 Pages

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01486v1) [paper-pdf](http://arxiv.org/pdf/2502.01486v1)

**Authors**: Suryansh Upadhyay, Swaroop Ghosh

**Abstract**: Quantum computing (QC) has the potential to revolutionize fields like machine learning, security, and healthcare. Quantum machine learning (QML) has emerged as a promising area, enhancing learning algorithms using quantum computers. However, QML models are lucrative targets due to their high training costs and extensive training times. The scarcity of quantum resources and long wait times further exacerbate the challenge. Additionally, QML providers may rely on third party quantum clouds for hosting models, exposing them and their training data to potential threats. As QML as a Service (QMLaaS) becomes more prevalent, reliance on third party quantum clouds poses a significant security risk. This work demonstrates that adversaries in quantum cloud environments can exploit white box access to QML models to infer the users encoding scheme by analyzing circuit transpilation artifacts. The extracted data can be reused for training clone models or sold for profit. We validate the proposed attack through simulations, achieving high accuracy in distinguishing between encoding schemes. We report that 95% of the time, the encoding can be predicted correctly. To mitigate this threat, we propose a transient obfuscation layer that masks encoding fingerprints using randomized rotations and entanglement, reducing adversarial detection to near random chance 42% , with a depth overhead of 8.5% for a 5 layer QNN design.



## **14. DeTrigger: A Gradient-Centric Approach to Backdoor Attack Mitigation in Federated Learning**

cs.LG

21 pages

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2411.12220v2) [paper-pdf](http://arxiv.org/pdf/2411.12220v2)

**Authors**: Kichang Lee, Yujin Shin, Jonghyuk Yun, Songkuk Kim, Jun Han, JeongGil Ko

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed devices while preserving local data privacy, making it ideal for mobile and embedded systems. However, the decentralized nature of FL also opens vulnerabilities to model poisoning attacks, particularly backdoor attacks, where adversaries implant trigger patterns to manipulate model predictions. In this paper, we propose DeTrigger, a scalable and efficient backdoor-robust federated learning framework that leverages insights from adversarial attack methodologies. By employing gradient analysis with temperature scaling, DeTrigger detects and isolates backdoor triggers, allowing for precise model weight pruning of backdoor activations without sacrificing benign model knowledge. Extensive evaluations across four widely used datasets demonstrate that DeTrigger achieves up to 251x faster detection than traditional methods and mitigates backdoor attacks by up to 98.9%, with minimal impact on global model accuracy. Our findings establish DeTrigger as a robust and scalable solution to protect federated learning environments against sophisticated backdoor threats.



## **15. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

cs.CL

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01386v1) [paper-pdf](http://arxiv.org/pdf/2502.01386v1)

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.



## **16. Detecting Backdoor Samples in Contrastive Language Image Pretraining**

cs.LG

ICLR2025

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01385v1) [paper-pdf](http://arxiv.org/pdf/2502.01385v1)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: Contrastive language-image pretraining (CLIP) has been found to be vulnerable to poisoning backdoor attacks where the adversary can achieve an almost perfect attack success rate on CLIP models by poisoning only 0.01\% of the training dataset. This raises security concerns on the current practice of pretraining large-scale models on unscrutinized web data using CLIP. In this work, we analyze the representations of backdoor-poisoned samples learned by CLIP models and find that they exhibit unique characteristics in their local subspace, i.e., their local neighborhoods are far more sparse than that of clean samples. Based on this finding, we conduct a systematic study on detecting CLIP backdoor attacks and show that these attacks can be easily and efficiently detected by traditional density ratio-based local outlier detectors, whereas existing backdoor sample detection methods fail. Our experiments also reveal that an unintentional backdoor already exists in the original CC3M dataset and has been trained into a popular open-source model released by OpenCLIP. Based on our detector, one can clean up a million-scale web dataset (e.g., CC3M) efficiently within 15 minutes using 4 Nvidia A100 GPUs. The code is publicly available in our \href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub repository}.



## **17. Improving the Robustness of Representation Misdirection for Large Language Model Unlearning**

cs.CL

12 pages, 4 figures, 1 table

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2501.19202v2) [paper-pdf](http://arxiv.org/pdf/2501.19202v2)

**Authors**: Dang Huu-Tien, Hoang Thanh-Tung, Le-Minh Nguyen, Naoya Inoue

**Abstract**: Representation Misdirection (RM) and variants are established large language model (LLM) unlearning methods with state-of-the-art performance. In this paper, we show that RM methods inherently reduce models' robustness, causing them to misbehave even when a single non-adversarial forget-token is in the retain-query. Toward understanding underlying causes, we reframe the unlearning process as backdoor attacks and defenses: forget-tokens act as backdoor triggers that, when activated in retain-queries, cause disruptions in RM models' behaviors, similar to successful backdoor attacks. To mitigate this vulnerability, we propose Random Noise Augmentation -- a model and method agnostic approach with theoretical guarantees for improving the robustness of RM methods. Extensive experiments demonstrate that RNA significantly improves the robustness of RM models while enhancing the unlearning performances.



## **18. Unified Breakdown Analysis for Byzantine Robust Gossip**

math.OC

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2410.10418v2) [paper-pdf](http://arxiv.org/pdf/2410.10418v2)

**Authors**: Renaud Gaucher, Aymeric Dieuleveut, Hadrien Hendrikx

**Abstract**: In decentralized machine learning, different devices communicate in a peer-to-peer manner to collaboratively learn from each other's data. Such approaches are vulnerable to misbehaving (or Byzantine) devices. We introduce $\mathrm{F}\text{-}\rm RG$, a general framework for building robust decentralized algorithms with guarantees arising from robust-sum-like aggregation rules $\mathrm{F}$. We then investigate the notion of *breakdown point*, and show an upper bound on the number of adversaries that decentralized algorithms can tolerate. We introduce a practical robust aggregation rule, coined $\rm CS_{ours}$, such that $\rm CS_{ours}\text{-}RG$ has a near-optimal breakdown. Other choices of aggregation rules lead to existing algorithms such as $\rm ClippedGossip$ or $\rm NNA$. We give experimental evidence to validate the effectiveness of $\rm CS_{ours}\text{-}RG$ and highlight the gap with $\mathrm{NNA}$, in particular against a novel attack tailored to decentralized communications.



## **19. FSPGD: Rethinking Black-box Attacks on Semantic Segmentation**

cs.CV

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01262v1) [paper-pdf](http://arxiv.org/pdf/2502.01262v1)

**Authors**: Eun-Sol Park, MiSo Park, Seung Park, Yong-Goo Shin

**Abstract**: Transferability, the ability of adversarial examples crafted for one model to deceive other models, is crucial for black-box attacks. Despite advancements in attack methods for semantic segmentation, transferability remains limited, reducing their effectiveness in real-world applications. To address this, we introduce the Feature Similarity Projected Gradient Descent (FSPGD) attack, a novel black-box approach that enhances both attack performance and transferability. Unlike conventional segmentation attacks that rely on output predictions for gradient calculation, FSPGD computes gradients from intermediate layer features. Specifically, our method introduces a loss function that targets local information by comparing features between clean images and adversarial examples, while also disrupting contextual information by accounting for spatial relationships between objects. Experiments on Pascal VOC 2012 and Cityscapes datasets demonstrate that FSPGD achieves superior transferability and attack performance, establishing a new state-of-the-art benchmark. Code is available at https://github.com/KU-AIVS/FSPGD.



## **20. The Impact of Logic Locking on Confidentiality: An Automated Evaluation**

cs.CR

8 pages, accepted at 26th International Symposium on Quality  Electronic Design (ISQED'25)

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01240v1) [paper-pdf](http://arxiv.org/pdf/2502.01240v1)

**Authors**: Lennart M. Reimann, Evgenii Rezunov, Dominik Germek, Luca Collini, Christian Pilato, Ramesh Karri, Rainer Leupers

**Abstract**: Logic locking secures hardware designs in untrusted foundries by incorporating key-driven gates to obscure the original blueprint. While this method safeguards the integrated circuit from malicious alterations during fabrication, its influence on data confidentiality during runtime has been ignored. In this study, we employ path sensitization to formally examine the impact of logic locking on confidentiality. By applying three representative logic locking mechanisms on open-source cryptographic benchmarks, we utilize an automatic test pattern generation framework to evaluate the effect of locking on cryptographic encryption keys and sensitive data signals. Our analysis reveals that logic locking can inadvertently cause sensitive data leakage when incorrect logic locking keys are used. We show that a single malicious logic locking key can expose over 70% of an encryption key. If an adversary gains control over other inputs, the entire encryption key can be compromised. This research uncovers a significant security vulnerability in logic locking and emphasizes the need for comprehensive security assessments that extend beyond key-recovery attacks.



## **21. The dark deep side of DeepSeek: Fine-tuning attacks against the safety alignment of CoT-enabled models**

cs.CR

12 Pages

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01225v1) [paper-pdf](http://arxiv.org/pdf/2502.01225v1)

**Authors**: Zhiyuan Xu, Joseph Gardiner, Sana Belguith

**Abstract**: Large language models are typically trained on vast amounts of data during the pre-training phase, which may include some potentially harmful information. Fine-tuning attacks can exploit this by prompting the model to reveal such behaviours, leading to the generation of harmful content. In this paper, we focus on investigating the performance of the Chain of Thought based reasoning model, DeepSeek, when subjected to fine-tuning attacks. Specifically, we explore how fine-tuning manipulates the model's output, exacerbating the harmfulness of its responses while examining the interaction between the Chain of Thought reasoning and adversarial inputs. Through this study, we aim to shed light on the vulnerability of Chain of Thought enabled models to fine-tuning attacks and the implications for their safety and ethical deployment.



## **22. Jailbreaking with Universal Multi-Prompts**

cs.CL

Accepted by NAACL Findings 2025

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01154v1) [paper-pdf](http://arxiv.org/pdf/2502.01154v1)

**Authors**: Yu-Ling Hsu, Hsuan Su, Shang-Tse Chen

**Abstract**: Large language models (LLMs) have seen rapid development in recent years, revolutionizing various applications and significantly enhancing convenience and productivity. However, alongside their impressive capabilities, ethical concerns and new types of attacks, such as jailbreaking, have emerged. While most prompting techniques focus on optimizing adversarial inputs for individual cases, resulting in higher computational costs when dealing with large datasets. Less research has addressed the more general setting of training a universal attacker that can transfer to unseen tasks. In this paper, we introduce JUMP, a prompt-based method designed to jailbreak LLMs using universal multi-prompts. We also adapt our approach for defense, which we term DUMP. Experimental results demonstrate that our method for optimizing universal multi-prompts outperforms existing techniques.



## **23. Warfare:Breaking the Watermark Protection of AI-Generated Content**

cs.CV

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2310.07726v4) [paper-pdf](http://arxiv.org/pdf/2310.07726v4)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Shangwei Guo, Han Qiu, Guoyin Wang, Jiwei Li, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is rapidly expanding, with services using advanced generative models to create realistic images and fluent text. Regulating such content is crucial to prevent policy violations, such as unauthorized commercialization or unsafe content distribution. Watermarking is a promising solution for content attribution and verification, but we demonstrate its vulnerability to two key attacks: (1) Watermark removal, where adversaries erase embedded marks to evade regulation, and (2) Watermark forging, where they generate illicit content with forged watermarks, leading to misattribution. We propose Warfare, a unified attack framework leveraging a pre-trained diffusion model for content processing and a generative adversarial network for watermark manipulation. Evaluations across datasets and embedding setups show that Warfare achieves high success rates while preserving content quality. We further introduce Warfare-Plus, which enhances efficiency without compromising effectiveness. The code can be found in https://github.com/GuanlinLee/warfare.



## **24. Adversarial Robustness in Two-Stage Learning-to-Defer: Algorithms and Guarantees**

stat.ML

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01027v1) [paper-pdf](http://arxiv.org/pdf/2502.01027v1)

**Authors**: Yannis Montreuil, Axel Carlier, Lai Xing Ng, Wei Tsang Ooi

**Abstract**: Learning-to-Defer (L2D) facilitates optimal task allocation between AI systems and decision-makers. Despite its potential, we show that current two-stage L2D frameworks are highly vulnerable to adversarial attacks, which can misdirect queries or overwhelm decision agents, significantly degrading system performance. This paper conducts the first comprehensive analysis of adversarial robustness in two-stage L2D frameworks. We introduce two novel attack strategies -- untargeted and targeted -- that exploit inherent structural vulnerabilities in these systems. To mitigate these threats, we propose SARD, a robust, convex, deferral algorithm rooted in Bayes and $(\mathcal{R},\mathcal{G})$-consistency. Our approach guarantees optimal task allocation under adversarial perturbations for all surrogates in the cross-entropy family. Extensive experiments on classification, regression, and multi-task benchmarks validate the robustness of SARD.



## **25. Refining Adaptive Zeroth-Order Optimization at Ease**

cs.LG

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01014v1) [paper-pdf](http://arxiv.org/pdf/2502.01014v1)

**Authors**: Yao Shu, Qixin Zhang, Kun He, Zhongxiang Dai

**Abstract**: Recently, zeroth-order (ZO) optimization plays an essential role in scenarios where gradient information is inaccessible or unaffordable, such as black-box systems and resource-constrained environments. While existing adaptive methods such as ZO-AdaMM have shown promise, they are fundamentally limited by their underutilization of moment information during optimization, usually resulting in underperforming convergence. To overcome these limitations, this paper introduces Refined Adaptive Zeroth-Order Optimization (R-AdaZO). Specifically, we first show the untapped variance reduction effect of first moment estimate on ZO gradient estimation, which improves the accuracy and stability of ZO updates. We then refine the second moment estimate based on these variance-reduced gradient estimates to better capture the geometry of the optimization landscape, enabling a more effective scaling of ZO updates. We present rigorous theoretical analysis to show (I) the first analysis to the variance reduction of first moment estimate in ZO optimization, (II) the improved second moment estimates with a more accurate approximation of its variance-free ideal, (III) the first variance-aware convergence framework for adaptive ZO methods, which may be of independent interest, and (IV) the faster convergence of R-AdaZO than existing baselines like ZO-AdaMM. Our extensive experiments, including synthetic problems, black-box adversarial attack, and memory-efficient fine-tuning of large language models (LLMs), further verify the superior convergence of R-AdaZO, indicating that R-AdaZO offers an improved solution for real-world ZO optimization challenges.



## **26. Boosting Adversarial Robustness and Generalization with Structural Prior**

cs.LG

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00834v1) [paper-pdf](http://arxiv.org/pdf/2502.00834v1)

**Authors**: Zhichao Hou, Weizhi Gao, Hamid Krim, Xiaorui Liu

**Abstract**: This work investigates a novel approach to boost adversarial robustness and generalization by incorporating structural prior into the design of deep learning models. Specifically, our study surprisingly reveals that existing dictionary learning-inspired convolutional neural networks (CNNs) provide a false sense of security against adversarial attacks. To address this, we propose Elastic Dictionary Learning Networks (EDLNets), a novel ResNet architecture that significantly enhances adversarial robustness and generalization. This novel and effective approach is supported by a theoretical robustness analysis using influence functions. Moreover, extensive and reliable experiments demonstrate consistent and significant performance improvement on open robustness leaderboards such as RobustBench, surpassing state-of-the-art baselines. To the best of our knowledge, this is the first work to discover and validate that structural prior can reliably enhance deep learning robustness under strong adaptive attacks, unveiling a promising direction for future research.



## **27. AGNNCert: Defending Graph Neural Networks against Arbitrary Perturbations with Deterministic Certification**

cs.CR

Accepted by Usenix Security 2025

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00765v1) [paper-pdf](http://arxiv.org/pdf/2502.00765v1)

**Authors**: Jiate Li, Binghui Wang

**Abstract**: Graph neural networks (GNNs) achieve the state-of-the-art on graph-relevant tasks such as node and graph classification. However, recent works show GNNs are vulnerable to adversarial perturbations include the perturbation on edges, nodes, and node features, the three components forming a graph. Empirical defenses against such attacks are soon broken by adaptive ones. While certified defenses offer robustness guarantees, they face several limitations: 1) almost all restrict the adversary's capability to only one type of perturbation, which is impractical; 2) all are designed for a particular GNN task, which limits their applicability; and 3) the robustness guarantees of all methods except one are not 100% accurate.   We address all these limitations by developing AGNNCert, the first certified defense for GNNs against arbitrary (edge, node, and node feature) perturbations with deterministic robustness guarantees, and applicable to the two most common node and graph classification tasks. AGNNCert also encompass existing certified defenses as special cases. Extensive evaluations on multiple benchmark node/graph classification datasets and two real-world graph datasets, and multiple GNNs validate the effectiveness of AGNNCert to provably defend against arbitrary perturbations. AGNNCert also shows its superiority over the state-of-the-art certified defenses against the individual edge perturbation and node perturbation.



## **28. Decentralized Nonconvex Robust Optimization over Unsafe Multiagent Systems: System Modeling, Utility, Resilience, and Privacy Analysis**

math.OC

15 pages, 15 figures

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2409.18632v6) [paper-pdf](http://arxiv.org/pdf/2409.18632v6)

**Authors**: Jinhui Hu, Guo Chen, Huaqing Li, Huqiang Cheng, Xiaoyu Guo, Tingwen Huang

**Abstract**: Privacy leakage and Byzantine failures are two adverse factors to the intelligent decision-making process of multi-agent systems (MASs). Considering the presence of these two issues, this paper targets the resolution of a class of nonconvex optimization problems under the Polyak-{\L}ojasiewicz (P-{\L}) condition. To address this problem, we first identify and construct the adversary system model. To enhance the robustness of stochastic gradient descent methods, we mask the local gradients with Gaussian noises and adopt a resilient aggregation method self-centered clipping (SCC) to design a differentially private (DP) decentralized Byzantine-resilient algorithm, namely DP-SCC-PL, which simultaneously achieves differential privacy and Byzantine resilience. The convergence analysis of DP-SCC-PL is challenging since the convergence error can be contributed jointly by privacy-preserving and Byzantine-resilient mechanisms, as well as the nonconvex relaxation, which is addressed via seeking the contraction relationships among the disagreement measure of reliable agents before and after aggregation, together with the optimal gap. Theoretical results reveal that DP-SCC-PL achieves consensus among all reliable agents and sublinear (inexact) convergence with well-designed step-sizes. It has also been proved that if there are no privacy issues and Byzantine agents, then the asymptotic exact convergence can be recovered. Numerical experiments verify the utility, resilience, and differential privacy of DP-SCC-PL by tackling a nonconvex optimization problem satisfying the P-{\L} condition under various Byzantine attacks.



## **29. Gandalf the Red: Adaptive Security for LLMs**

cs.LG

Niklas Pfister, V\'aclav Volhejn and Manuel Knott contributed equally

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2501.07927v2) [paper-pdf](http://arxiv.org/pdf/2501.07927v2)

**Authors**: Niklas Pfister, Václav Volhejn, Manuel Knott, Santiago Arias, Julia Bazińska, Mykhailo Bichurin, Alan Commike, Janet Darling, Peter Dienes, Matthew Fiedler, David Haber, Matthias Kraft, Marco Lancini, Max Mathys, Damián Pascual-Ortiz, Jakub Podolak, Adrià Romero-López, Kyriacos Shiarlis, Andreas Signer, Zsolt Terek, Athanasios Theocharis, Daniel Timbrell, Samuel Trautwein, Samuel Watts, Yun-Han Wu, Mateo Rojas-Carulla

**Abstract**: Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications.



## **30. From Compliance to Exploitation: Jailbreak Prompt Attacks on Multimodal LLMs**

cs.CR

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00735v1) [paper-pdf](http://arxiv.org/pdf/2502.00735v1)

**Authors**: Chun Wai Chiu, Linghan Huang, Bo Li, Huaming Chen

**Abstract**: Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the frontier multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. To better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flank Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios. These findings highlight both the potency of prompt-based obfuscation in voice-enabled contexts and the limitations of current LLMs' moderation safeguards and the urgent need for advanced defense strategies to address the challenges posed by evolving, context-rich attacks.



## **31. "I am bad": Interpreting Stealthy, Universal and Robust Audio Jailbreaks in Audio-Language Models**

cs.LG

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00718v1) [paper-pdf](http://arxiv.org/pdf/2502.00718v1)

**Authors**: Isha Gupta, David Khachaturov, Robert Mullins

**Abstract**: The rise of multimodal large language models has introduced innovative human-machine interaction paradigms but also significant challenges in machine learning safety. Audio-Language Models (ALMs) are especially relevant due to the intuitive nature of spoken communication, yet little is known about their failure modes. This paper explores audio jailbreaks targeting ALMs, focusing on their ability to bypass alignment mechanisms. We construct adversarial perturbations that generalize across prompts, tasks, and even base audio samples, demonstrating the first universal jailbreaks in the audio modality, and show that these remain effective in simulated real-world conditions. Beyond demonstrating attack feasibility, we analyze how ALMs interpret these audio adversarial examples and reveal them to encode imperceptible first-person toxic speech - suggesting that the most effective perturbations for eliciting toxic outputs specifically embed linguistic features within the audio signal. These results have important implications for understanding the interactions between different modalities in multimodal models, and offer actionable insights for enhancing defenses against adversarial audio attacks.



## **32. Transferable Adversarial Face Attack with Text Controlled Attribute**

cs.CV

Accepted by AAAI 2025

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2412.11735v2) [paper-pdf](http://arxiv.org/pdf/2412.11735v2)

**Authors**: Wenyun Li, Zheng Zhang, Xiangyuan Lan, Dongmei Jiang

**Abstract**: Traditional adversarial attacks typically produce adversarial examples under norm-constrained conditions, whereas unrestricted adversarial examples are free-form with semantically meaningful perturbations. Current unrestricted adversarial impersonation attacks exhibit limited control over adversarial face attributes and often suffer from low transferability. In this paper, we propose a novel Text Controlled Attribute Attack (TCA$^2$) to generate photorealistic adversarial impersonation faces guided by natural language. Specifically, the category-level personal softmax vector is employed to precisely guide the impersonation attacks. Additionally, we propose both data and model augmentation strategies to achieve transferable attacks on unknown target models. Finally, a generative model, \textit{i.e}, Style-GAN, is utilized to synthesize impersonated faces with desired attributes. Extensive experiments on two high-resolution face recognition datasets validate that our TCA$^2$ method can generate natural text-guided adversarial impersonation faces with high transferability. We also evaluate our method on real-world face recognition systems, \textit{i.e}, Face++ and Aliyun, further demonstrating the practical potential of our approach.



## **33. The Great Contradiction Showdown: How Jailbreak and Stealth Wrestle in Vision-Language Models?**

cs.LG

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2410.01438v2) [paper-pdf](http://arxiv.org/pdf/2410.01438v2)

**Authors**: Ching-Chia Kao, Chia-Mu Yu, Chun-Shien Lu, Chu-Song Chen

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable performance on a variety of tasks, yet they remain vulnerable to jailbreak attacks that compromise safety and reliability. In this paper, we provide an information-theoretic framework for understanding the fundamental trade-off between the effectiveness of these attacks and their stealthiness. Drawing on Fano's inequality, we demonstrate how an attacker's success probability is intrinsically linked to the stealthiness of generated prompts. Building on this, we propose an efficient algorithm for detecting non-stealthy jailbreak attacks, offering significant improvements in model robustness. Experimental results highlight the tension between strong attacks and their detectability, providing insights into both adversarial strategies and defense mechanisms.



## **34. Towards Robust Multimodal Large Language Models Against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00653v1) [paper-pdf](http://arxiv.org/pdf/2502.00653v1)

**Authors**: Ziyi Yin, Yuanpu Cao, Han Liu, Ting Wang, Jinghui Chen, Fenhlong Ma

**Abstract**: While multimodal large language models (MLLMs) have achieved remarkable success in recent advancements, their susceptibility to jailbreak attacks has come to light. In such attacks, adversaries exploit carefully crafted prompts to coerce models into generating harmful or undesirable content. Existing defense mechanisms often rely on external inference steps or safety alignment training, both of which are less effective and impractical when facing sophisticated adversarial perturbations in white-box scenarios. To address these challenges and bolster MLLM robustness, we introduce SafeMLLM by adopting an adversarial training framework that alternates between an attack step for generating adversarial noise and a model updating step. At the attack step, SafeMLLM generates adversarial perturbations through a newly proposed contrastive embedding attack (CoE-Attack), which optimizes token embeddings under a contrastive objective. SafeMLLM then updates model parameters to neutralize the perturbation effects while preserving model utility on benign inputs. We evaluate SafeMLLM across six MLLMs and six jailbreak methods spanning multiple modalities. Experimental results show that SafeMLLM effectively defends against diverse attacks, maintaining robust performance and utilities.



## **35. Reformulation is All You Need: Addressing Malicious Text Features in DNNs**

cs.LG

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00652v1) [paper-pdf](http://arxiv.org/pdf/2502.00652v1)

**Authors**: Yi Jiang, Oubo Ma, Yong Yang, Tong Zhang, Shouling Ji

**Abstract**: Human language encompasses a wide range of intricate and diverse implicit features, which attackers can exploit to launch adversarial or backdoor attacks, compromising DNN models for NLP tasks. Existing model-oriented defenses often require substantial computational resources as model size increases, whereas sample-oriented defenses typically focus on specific attack vectors or schemes, rendering them vulnerable to adaptive attacks. We observe that the root cause of both adversarial and backdoor attacks lies in the encoding process of DNN models, where subtle textual features, negligible for human comprehension, are erroneously assigned significant weight by less robust or trojaned models. Based on it we propose a unified and adaptive defense framework that is effective against both adversarial and backdoor attacks. Our approach leverages reformulation modules to address potential malicious features in textual inputs while preserving the original semantic integrity. Extensive experiments demonstrate that our framework outperforms existing sample-oriented defense baselines across a diverse range of malicious textual features.



## **36. TrojanTime: Backdoor Attacks on Time Series Classification**

cs.CR

13 pages, 3 figures, 3 tables

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00646v1) [paper-pdf](http://arxiv.org/pdf/2502.00646v1)

**Authors**: Chang Dong, Zechao Sun, Guangdong Bai, Shuying Piao, Weitong Chen, Wei Emma Zhang

**Abstract**: Time Series Classification (TSC) is highly vulnerable to backdoor attacks, posing significant security threats. Existing methods primarily focus on data poisoning during the training phase, designing sophisticated triggers to improve stealthiness and attack success rate (ASR). However, in practical scenarios, attackers often face restrictions in accessing training data. Moreover, it is a challenge for the model to maintain generalization ability on clean test data while remaining vulnerable to poisoned inputs when data is inaccessible. To address these challenges, we propose TrojanTime, a novel two-step training algorithm. In the first stage, we generate a pseudo-dataset using an external arbitrary dataset through target adversarial attacks. The clean model is then continually trained on this pseudo-dataset and its poisoned version. To ensure generalization ability, the second stage employs a carefully designed training strategy, combining logits alignment and batch norm freezing. We evaluate TrojanTime using five types of triggers across four TSC architectures in UCR benchmark datasets from diverse domains. The results demonstrate the effectiveness of TrojanTime in executing backdoor attacks while maintaining clean accuracy. Finally, to mitigate this threat, we propose a defensive unlearning strategy that effectively reduces the ASR while preserving clean accuracy.



## **37. Actor Critic with Experience Replay-based automatic treatment planning for prostate cancer intensity modulated radiotherapy**

cs.LG

27 Pages, 8 Figures, 4 Tables

**SubmitDate**: 2025-02-01    [abs](http://arxiv.org/abs/2502.00346v1) [paper-pdf](http://arxiv.org/pdf/2502.00346v1)

**Authors**: Md Mainul Abrar, Parvat Sapkota, Damon Sprouts, Xun Jia, Yujie Chi

**Abstract**: Background: Real-time treatment planning in IMRT is challenging due to complex beam interactions. AI has improved automation, but existing models require large, high-quality datasets and lack universal applicability. Deep reinforcement learning (DRL) offers a promising alternative by mimicking human trial-and-error planning.   Purpose: Develop a stochastic policy-based DRL agent for automatic treatment planning with efficient training, broad applicability, and robustness against adversarial attacks using Fast Gradient Sign Method (FGSM).   Methods: Using the Actor-Critic with Experience Replay (ACER) architecture, the agent tunes treatment planning parameters (TPPs) in inverse planning. Training is based on prostate cancer IMRT cases, using dose-volume histograms (DVHs) as input. The model is trained on a single patient case, validated on two independent cases, and tested on 300+ plans across three datasets. Plan quality is assessed using ProKnow scores, and robustness is tested against adversarial attacks.   Results: Despite training on a single case, the model generalizes well. Before ACER-based planning, the mean plan score was 6.20$\pm$1.84; after, 93.09% of cases achieved a perfect score of 9, with a mean of 8.93$\pm$0.27. The agent effectively prioritizes optimal TPP tuning and remains robust against adversarial attacks.   Conclusions: The ACER-based DRL agent enables efficient, high-quality treatment planning in prostate cancer IMRT, demonstrating strong generalizability and robustness.



## **38. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**

cs.CR

**SubmitDate**: 2025-02-01    [abs](http://arxiv.org/abs/2502.00306v1) [paper-pdf](http://arxiv.org/pdf/2502.00306v1)

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference.



## **39. Patch Synthesis for Property Repair of Deep Neural Networks**

cs.LG

**SubmitDate**: 2025-02-01    [abs](http://arxiv.org/abs/2404.01642v2) [paper-pdf](http://arxiv.org/pdf/2404.01642v2)

**Authors**: Zhiming Chi, Jianan Ma, Pengfei Yang, Cheng-Chao Huang, Renjue Li, Xiaowei Huang, Lijun Zhang

**Abstract**: Deep neural networks (DNNs) are prone to various dependability issues, such as adversarial attacks, which hinder their adoption in safety-critical domains. Recently, NN repair techniques have been proposed to address these issues while preserving original performance by locating and modifying guilty neurons and their parameters. However, existing repair approaches are often limited to specific data sets and do not provide theoretical guarantees for the effectiveness of the repairs. To address these limitations, we introduce PatchPro, a novel patch-based approach for property-level repair of DNNs, focusing on local robustness. The key idea behind PatchPro is to construct patch modules that, when integrated with the original network, provide specialized repairs for all samples within the robustness neighborhood while maintaining the network's original performance. Our method incorporates formal verification and a heuristic mechanism for allocating patch modules, enabling it to defend against adversarial attacks and generalize to other inputs. PatchPro demonstrates superior efficiency, scalability, and repair success rates compared to existing DNN repair methods, i.e., realizing provable property-level repair for 100% cases across multiple high-dimensional datasets.



## **40. Robustifying ML-powered Network Classifiers with PANTS**

cs.CR

**SubmitDate**: 2025-02-01    [abs](http://arxiv.org/abs/2409.04691v3) [paper-pdf](http://arxiv.org/pdf/2409.04691v3)

**Authors**: Minhao Jin, Maria Apostolaki

**Abstract**: Multiple network management tasks, from resource allocation to intrusion detection, rely on some form of ML-based network traffic classification (MNC). Despite their potential, MNCs are vulnerable to adversarial inputs, which can lead to outages, poor decision-making, and security violations, among other issues. The goal of this paper is to help network operators assess and enhance the robustness of their MNC against adversarial inputs. The most critical step for this is generating inputs that can fool the MNC while being realizable under various threat models. Compared to other ML models, finding adversarial inputs against MNCs is more challenging due to the existence of non-differentiable components e.g., traffic engineering and the need to constrain inputs to preserve semantics and ensure reliability. These factors prevent the direct use of well-established gradient-based methods developed in adversarial ML (AML). To address these challenges, we introduce PANTS, a practical white-box framework that uniquely integrates AML techniques with Satisfiability Modulo Theories (SMT) solvers to generate adversarial inputs for MNCs. We also embed PANTS into an iterative adversarial training process that enhances the robustness of MNCs against adversarial inputs. PANTS is 70% and 2x more likely in median to find adversarial inputs against target MNCs compared to state-of-the-art baselines, namely Amoeba and BAP. PANTS improves the robustness of the target MNCs by 52.7% (even against attackers outside of what is considered during robustification) without sacrificing their accuracy.



## **41. Redefining Machine Unlearning: A Conformal Prediction-Motivated Approach**

cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19403v1) [paper-pdf](http://arxiv.org/pdf/2501.19403v1)

**Authors**: Yingdan Shi, Ren Wang

**Abstract**: Machine unlearning seeks to systematically remove specified data from a trained model, effectively achieving a state as though the data had never been encountered during training. While metrics such as Unlearning Accuracy (UA) and Membership Inference Attack (MIA) provide a baseline for assessing unlearning performance, they fall short of evaluating the completeness and reliability of forgetting. This is because the ground truth labels remain potential candidates within the scope of uncertainty quantification, leaving gaps in the evaluation of true forgetting. In this paper, we identify critical limitations in existing unlearning metrics and propose enhanced evaluation metrics inspired by conformal prediction. Our metrics can effectively capture the extent to which ground truth labels are excluded from the prediction set. Furthermore, we observe that many existing machine unlearning methods do not achieve satisfactory forgetting performance when evaluated with our new metrics. To address this, we propose an unlearning framework that integrates conformal prediction insights into Carlini & Wagner adversarial attack loss. Extensive experiments on the image classification task demonstrate that our enhanced metrics offer deeper insights into unlearning effectiveness, and that our unlearning framework significantly improves the forgetting quality of unlearning methods.



## **42. SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders**

cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.18052v2) [paper-pdf](http://arxiv.org/pdf/2501.18052v2)

**Authors**: Bartosz Cywiński, Kamil Deja

**Abstract**: Diffusion models, while powerful, can inadvertently generate harmful or undesirable content, raising significant ethical and safety concerns. Recent machine unlearning approaches offer potential solutions but often lack transparency, making it difficult to understand the changes they introduce to the base model. In this work, we introduce SAeUron, a novel method leveraging features learned by sparse autoencoders (SAEs) to remove unwanted concepts in text-to-image diffusion models. First, we demonstrate that SAEs, trained in an unsupervised manner on activations from multiple denoising timesteps of the diffusion model, capture sparse and interpretable features corresponding to specific concepts. Building on this, we propose a feature selection method that enables precise interventions on model activations to block targeted content while preserving overall performance. Evaluation with the competitive UnlearnCanvas benchmark on object and style unlearning highlights SAeUron's state-of-the-art performance. Moreover, we show that with a single SAE, we can remove multiple concepts simultaneously and that in contrast to other methods, SAeUron mitigates the possibility of generating unwanted content, even under adversarial attack. Code and checkpoints are available at: https://github.com/cywinski/SAeUron.



## **43. UniGuard: Towards Universal Safety Guardrails for Jailbreak Attacks on Multimodal Large Language Models**

cs.CL

14 pages

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2411.01703v2) [paper-pdf](http://arxiv.org/pdf/2411.01703v2)

**Authors**: Sejoon Oh, Yiqiao Jin, Megha Sharma, Donghyun Kim, Eric Ma, Gaurav Verma, Srijan Kumar

**Abstract**: Multimodal large language models (MLLMs) have revolutionized vision-language understanding but remain vulnerable to multimodal jailbreak attacks, where adversarial inputs are meticulously crafted to elicit harmful or inappropriate responses. We propose UniGuard, a novel multimodal safety guardrail that jointly considers the unimodal and cross-modal harmful signals. UniGuard trains a multimodal guardrail to minimize the likelihood of generating harmful responses in a toxic corpus. The guardrail can be seamlessly applied to any input prompt during inference with minimal computational costs. Extensive experiments demonstrate the generalizability of UniGuard across multiple modalities, attack strategies, and multiple state-of-the-art MLLMs, including LLaVA, Gemini Pro, GPT-4o, MiniGPT-4, and InstructBLIP. Notably, this robust defense mechanism maintains the models' overall vision-language understanding capabilities.



## **44. An Empirical Game-Theoretic Analysis of Autonomous Cyber-Defence Agents**

cs.AI

21 pages, 17 figures, 10 tables

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19206v1) [paper-pdf](http://arxiv.org/pdf/2501.19206v1)

**Authors**: Gregory Palmer, Luke Swaby, Daniel J. B. Harrold, Matthew Stewart, Alex Hiles, Chris Willis, Ian Miles, Sara Farmer

**Abstract**: The recent rise in increasingly sophisticated cyber-attacks raises the need for robust and resilient autonomous cyber-defence (ACD) agents. Given the variety of cyber-attack tactics, techniques and procedures (TTPs) employed, learning approaches that can return generalisable policies are desirable. Meanwhile, the assurance of ACD agents remains an open challenge. We address both challenges via an empirical game-theoretic analysis of deep reinforcement learning (DRL) approaches for ACD using the principled double oracle (DO) algorithm. This algorithm relies on adversaries iteratively learning (approximate) best responses against each others' policies; a computationally expensive endeavour for autonomous cyber operations agents. In this work we introduce and evaluate a theoretically-sound, potential-based reward shaping approach to expedite this process. In addition, given the increasing number of open-source ACD-DRL approaches, we extend the DO formulation to allow for multiple response oracles (MRO), providing a framework for a holistic evaluation of ACD approaches.



## **45. Enhancing Model Defense Against Jailbreaks with Proactive Safety Reasoning**

cs.CR

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19180v1) [paper-pdf](http://arxiv.org/pdf/2501.19180v1)

**Authors**: Xianglin Yang, Gelei Deng, Jieming Shi, Tianwei Zhang, Jin Song Dong

**Abstract**: Large language models (LLMs) are vital for a wide range of applications yet remain susceptible to jailbreak threats, which could lead to the generation of inappropriate responses. Conventional defenses, such as refusal and adversarial training, often fail to cover corner cases or rare domains, leaving LLMs still vulnerable to more sophisticated attacks. We propose a novel defense strategy, Safety Chain-of-Thought (SCoT), which harnesses the enhanced \textit{reasoning capabilities} of LLMs for proactive assessment of harmful inputs, rather than simply blocking them. SCoT augments any refusal training datasets to critically analyze the intent behind each request before generating answers. By employing proactive reasoning, SCoT enhances the generalization of LLMs across varied harmful queries and scenarios not covered in the safety alignment corpus. Additionally, it generates detailed refusals specifying the rules violated. Comparative evaluations show that SCoT significantly surpasses existing defenses, reducing vulnerability to out-of-distribution issues and adversarial manipulations while maintaining strong general capabilities.



## **46. Imitation Game for Adversarial Disillusion with Multimodal Generative Chain-of-Thought Role-Play**

cs.AI

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19143v1) [paper-pdf](http://arxiv.org/pdf/2501.19143v1)

**Authors**: Ching-Chun Chang, Fan-Yun Chen, Shih-Hong Gu, Kai Gao, Hanrui Wang, Isao Echizen

**Abstract**: As the cornerstone of artificial intelligence, machine perception confronts a fundamental threat posed by adversarial illusions. These adversarial attacks manifest in two primary forms: deductive illusion, where specific stimuli are crafted based on the victim model's general decision logic, and inductive illusion, where the victim model's general decision logic is shaped by specific stimuli. The former exploits the model's decision boundaries to create a stimulus that, when applied, interferes with its decision-making process. The latter reinforces a conditioned reflex in the model, embedding a backdoor during its learning phase that, when triggered by a stimulus, causes aberrant behaviours. The multifaceted nature of adversarial illusions calls for a unified defence framework, addressing vulnerabilities across various forms of attack. In this study, we propose a disillusion paradigm based on the concept of an imitation game. At the heart of the imitation game lies a multimodal generative agent, steered by chain-of-thought reasoning, which observes, internalises and reconstructs the semantic essence of a sample, liberated from the classic pursuit of reversing the sample to its original state. As a proof of concept, we conduct experimental simulations using a multimodal generative dialogue agent and evaluates the methodology under a variety of attack scenarios.



## **47. Average Certified Radius is a Poor Metric for Randomized Smoothing**

cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2410.06895v2) [paper-pdf](http://arxiv.org/pdf/2410.06895v2)

**Authors**: Chenhao Sun, Yuhao Mao, Mark Niklas Müller, Martin Vechev

**Abstract**: Randomized smoothing is a popular approach for providing certified robustness guarantees against adversarial attacks, and has become an active area of research. Over the past years, the average certified radius (ACR) has emerged as the most important metric for comparing methods and tracking progress in the field. However, in this work, for the first time we show that ACR is a poor metric for evaluating robustness guarantees provided by randomized smoothing. We theoretically prove not only that a trivial classifier can have arbitrarily large ACR, but also that ACR is much more sensitive to improvements on easy samples than on hard ones. Empirically, we confirm that existing training strategies, though improving ACR with different approaches, reduce the model's robustness on hard samples consistently. To strengthen our conclusion, we propose strategies, including explicitly discarding hard samples, reweighting the dataset with approximate certified radius, and extreme optimization for easy samples, to achieve state-of-the-art ACR, without training for robustness on the full data distribution. Overall, our results suggest that ACR has introduced a strong undesired bias to the field, and its application should be discontinued when evaluating randomized smoothing.



## **48. Understanding Oversmoothing in GNNs as Consensus in Opinion Dynamics**

cs.LG

23 pages, 3 figures

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19089v1) [paper-pdf](http://arxiv.org/pdf/2501.19089v1)

**Authors**: Keqin Wang, Yulong Yang, Ishan Saha, Christine Allen-Blanchette

**Abstract**: In contrast to classes of neural networks where the learned representations become increasingly expressive with network depth, the learned representations in graph neural networks (GNNs), tend to become increasingly similar. This phenomena, known as oversmoothing, is characterized by learned representations that cannot be reliably differentiated leading to reduced predictive performance. In this paper, we propose an analogy between oversmoothing in GNNs and consensus or agreement in opinion dynamics. Through this analogy, we show that the message passing structure of recent continuous-depth GNNs is equivalent to a special case of opinion dynamics (i.e., linear consensus models) which has been theoretically proven to converge to consensus (i.e., oversmoothing) for all inputs. Using the understanding developed through this analogy, we design a new continuous-depth GNN model based on nonlinear opinion dynamics and prove that our model, which we call behavior-inspired message passing neural network (BIMP) circumvents oversmoothing for general inputs. Through extensive experiments, we show that BIMP is robust to oversmoothing and adversarial attack, and consistently outperforms competitive baselines on numerous benchmarks.



## **49. Concept Steerers: Leveraging K-Sparse Autoencoders for Controllable Generations**

cs.CV

15 pages, 16 figures

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19066v1) [paper-pdf](http://arxiv.org/pdf/2501.19066v1)

**Authors**: Dahye Kim, Deepti Ghadiyaram

**Abstract**: Despite the remarkable progress in text-to-image generative models, they are prone to adversarial attacks and inadvertently generate unsafe, unethical content. Existing approaches often rely on fine-tuning models to remove specific concepts, which is computationally expensive, lack scalability, and/or compromise generation quality. In this work, we propose a novel framework leveraging k-sparse autoencoders (k-SAEs) to enable efficient and interpretable concept manipulation in diffusion models. Specifically, we first identify interpretable monosemantic concepts in the latent space of text embeddings and leverage them to precisely steer the generation away or towards a given concept (e.g., nudity) or to introduce a new concept (e.g., photographic style). Through extensive experiments, we demonstrate that our approach is very simple, requires no retraining of the base model nor LoRA adapters, does not compromise the generation quality, and is robust to adversarial prompt manipulations. Our method yields an improvement of $\mathbf{20.01\%}$ in unsafe concept removal, is effective in style manipulation, and is $\mathbf{\sim5}$x faster than current state-of-the-art.



## **50. Towards the Worst-case Robustness of Large Language Models**

cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19040v1) [paper-pdf](http://arxiv.org/pdf/2501.19040v1)

**Authors**: Huanran Chen, Yinpeng Dong, Zeming Wei, Hang Su, Jun Zhu

**Abstract**: Recent studies have revealed the vulnerability of Large Language Models (LLMs) to adversarial attacks, where the adversary crafts specific input sequences to induce harmful, violent, private, or incorrect outputs. Although various defenses have been proposed, they have not been evaluated by strong adaptive attacks, leaving the worst-case robustness of LLMs still intractable. By developing a stronger white-box attack, our evaluation results indicate that most typical defenses achieve nearly 0\% robustness.To solve this, we propose \textit{DiffTextPure}, a general defense that diffuses the (adversarial) input prompt using any pre-defined smoothing distribution, and purifies the diffused input using a pre-trained language model. Theoretically, we derive tight robustness lower bounds for all smoothing distributions using Fractal Knapsack or 0-1 Knapsack solvers. Under this framework, we certify the robustness of a specific case -- smoothing LLMs using a uniform kernel -- against \textit{any possible attack} with an average $\ell_0$ perturbation of 2.02 or an average suffix length of 6.41.



