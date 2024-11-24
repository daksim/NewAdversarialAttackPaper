# Latest Adversarial Attack Papers
**update at 2024-11-24 12:16:03**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Learning Fair Robustness via Domain Mixup**

cs.LG

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14424v1) [paper-pdf](http://arxiv.org/pdf/2411.14424v1)

**Authors**: Meiyu Zhong, Ravi Tandon

**Abstract**: Adversarial training is one of the predominant techniques for training classifiers that are robust to adversarial attacks. Recent work, however has found that adversarial training, which makes the overall classifier robust, it does not necessarily provide equal amount of robustness for all classes. In this paper, we propose the use of mixup for the problem of learning fair robust classifiers, which can provide similar robustness across all classes. Specifically, the idea is to mix inputs from the same classes and perform adversarial training on mixed up inputs. We present a theoretical analysis of this idea for the case of linear classifiers and show that mixup combined with adversarial training can provably reduce the class-wise robustness disparity. This method not only contributes to reducing the disparity in class-wise adversarial risk, but also the class-wise natural risk. Complementing our theoretical analysis, we also provide experimental results on both synthetic data and the real world dataset (CIFAR-10), which shows improvement in class wise disparities for both natural and adversarial risks.



## **2. Adversarial Poisoning Attack on Quantum Machine Learning Models**

quant-ph

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14412v1) [paper-pdf](http://arxiv.org/pdf/2411.14412v1)

**Authors**: Satwik Kundu, Swaroop Ghosh

**Abstract**: With the growing interest in Quantum Machine Learning (QML) and the increasing availability of quantum computers through cloud providers, addressing the potential security risks associated with QML has become an urgent priority. One key concern in the QML domain is the threat of data poisoning attacks in the current quantum cloud setting. Adversarial access to training data could severely compromise the integrity and availability of QML models. Classical data poisoning techniques require significant knowledge and training to generate poisoned data, and lack noise resilience, making them ineffective for QML models in the Noisy Intermediate Scale Quantum (NISQ) era. In this work, we first propose a simple yet effective technique to measure intra-class encoder state similarity (ESS) by analyzing the outputs of encoding circuits. Leveraging this approach, we introduce a quantum indiscriminate data poisoning attack, QUID. Through extensive experiments conducted in both noiseless and noisy environments (e.g., IBM\_Brisbane's noise), across various architectures and datasets, QUID achieves up to $92\%$ accuracy degradation in model performance compared to baseline models and up to $75\%$ accuracy degradation compared to random label-flipping. We also tested QUID against state-of-the-art classical defenses, with accuracy degradation still exceeding $50\%$, demonstrating its effectiveness. This work represents the first attempt to reevaluate data poisoning attacks in the context of QML.



## **3. Indiscriminate Disruption of Conditional Inference on Multivariate Gaussians**

stat.ML

30 pages, 6 figures; 4 tables

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14351v1) [paper-pdf](http://arxiv.org/pdf/2411.14351v1)

**Authors**: William N. Caballero, Matthew LaRosa, Alexander Fisher, Vahid Tarokh

**Abstract**: The multivariate Gaussian distribution underpins myriad operations-research, decision-analytic, and machine-learning models (e.g., Bayesian optimization, Gaussian influence diagrams, and variational autoencoders). However, despite recent advances in adversarial machine learning (AML), inference for Gaussian models in the presence of an adversary is notably understudied. Therefore, we consider a self-interested attacker who wishes to disrupt a decisionmaker's conditional inference and subsequent actions by corrupting a set of evidentiary variables. To avoid detection, the attacker also desires the attack to appear plausible wherein plausibility is determined by the density of the corrupted evidence. We consider white- and grey-box settings such that the attacker has complete and incomplete knowledge about the decisionmaker's underlying multivariate Gaussian distribution, respectively. Select instances are shown to reduce to quadratic and stochastic quadratic programs, and structural properties are derived to inform solution methods. We assess the impact and efficacy of these attacks in three examples, including, real estate evaluation, interest rate estimation and signals processing. Each example leverages an alternative underlying model, thereby highlighting the attacks' broad applicability. Through these applications, we also juxtapose the behavior of the white- and grey-box attacks to understand how uncertainty and structure affect attacker behavior.



## **4. Layer Pruning with Consensus: A Triple-Win Solution**

cs.LG

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14345v1) [paper-pdf](http://arxiv.org/pdf/2411.14345v1)

**Authors**: Leandro Giusti Mugnaini, Carolina Tavares Duarte, Anna H. Reali Costa, Artur Jordao

**Abstract**: Layer pruning offers a promising alternative to standard structured pruning, effectively reducing computational costs, latency, and memory footprint. While notable layer-pruning approaches aim to detect unimportant layers for removal, they often rely on single criteria that may not fully capture the complex, underlying properties of layers. We propose a novel approach that combines multiple similarity metrics into a single expressive measure of low-importance layers, called the Consensus criterion. Our technique delivers a triple-win solution: low accuracy drop, high-performance improvement, and increased robustness to adversarial attacks. With up to 78.80% FLOPs reduction and performance on par with state-of-the-art methods across different benchmarks, our approach reduces energy consumption and carbon emissions by up to 66.99% and 68.75%, respectively. Additionally, it avoids shortcut learning and improves robustness by up to 4 percentage points under various adversarial attacks. Overall, the Consensus criterion demonstrates its effectiveness in creating robust, efficient, and environmentally friendly pruned models.



## **5. Generating Realistic Adversarial Examples for Business Processes using Variational Autoencoders**

cs.LG

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14263v1) [paper-pdf](http://arxiv.org/pdf/2411.14263v1)

**Authors**: Alexander Stevens, Jari Peeperkorn, Johannes De Smedt, Jochen De Weerdt

**Abstract**: In predictive process monitoring, predictive models are vulnerable to adversarial attacks, where input perturbations can lead to incorrect predictions. Unlike in computer vision, where these perturbations are designed to be imperceptible to the human eye, the generation of adversarial examples in predictive process monitoring poses unique challenges. Minor changes to the activity sequences can create improbable or even impossible scenarios to occur due to underlying constraints such as regulatory rules or process constraints. To address this, we focus on generating realistic adversarial examples tailored to the business process context, in contrast to the imperceptible, pixel-level changes commonly seen in computer vision adversarial attacks. This paper introduces two novel latent space attacks, which generate adversaries by adding noise to the latent space representation of the input data, rather than directly modifying the input attributes. These latent space methods are domain-agnostic and do not rely on process-specific knowledge, as we restrict the generation of adversarial examples to the learned class-specific data distributions by directly perturbing the latent space representation of the business process executions. We evaluate these two latent space methods with six other adversarial attacking methods on eleven real-life event logs and four predictive models. The first three attacking methods directly permute the activities of the historically observed business process executions. The fourth method constrains the adversarial examples to lie within the same data distribution as the original instances, by projecting the adversarial examples to the original data distribution.



## **6. AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection**

cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14243v1) [paper-pdf](http://arxiv.org/pdf/2411.14243v1)

**Authors**: Jialin Lu, Junjie Shan, Ziqi Zhao, Ka-Ho Chow

**Abstract**: As object detection becomes integral to many safety-critical applications, understanding its vulnerabilities is essential. Backdoor attacks, in particular, pose a significant threat by implanting hidden backdoor in a victim model, which adversaries can later exploit to trigger malicious behaviors during inference. However, current backdoor techniques are limited to static scenarios where attackers must define a malicious objective before training, locking the attack into a predetermined action without inference-time adaptability. Given the expressive output space in object detection, including object existence detection, bounding box estimation, and object classification, the feasibility of implanting a backdoor that provides inference-time control with a high degree of freedom remains unexplored. This paper introduces AnywhereDoor, a flexible backdoor attack tailored for object detection. Once implanted, AnywhereDoor enables adversaries to specify different attack types (object vanishing, fabrication, or misclassification) and configurations (untargeted or targeted with specific classes) to dynamically control detection behavior. This flexibility is achieved through three key innovations: (i) objective disentanglement to support a broader range of attack combinations well beyond what existing methods allow; (ii) trigger mosaicking to ensure backdoor activations are robust, even against those object detectors that extract localized regions from the input image for recognition; and (iii) strategic batching to address object-level data imbalances that otherwise hinders a balanced manipulation. Extensive experiments demonstrate that AnywhereDoor provides attackers with a high degree of control, achieving an attack success rate improvement of nearly 80% compared to adaptations of existing methods for such flexible control.



## **7. GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs**

cs.LG

28 pages, 9 tables, 13 figures; under review at CVPR '25

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14133v1) [paper-pdf](http://arxiv.org/pdf/2411.14133v1)

**Authors**: Advik Raj Basani, Xiao Zhang

**Abstract**: Large Language Models (LLMs) have shown impressive proficiency across a range of natural language processing tasks yet remain vulnerable to adversarial prompts, known as jailbreak attacks, carefully designed to elicit harmful responses from LLMs. Traditional methods rely on manual heuristics, which suffer from limited generalizability. While being automatic, optimization-based attacks often produce unnatural jailbreak prompts that are easy to detect by safety filters or require high computational overhead due to discrete token optimization. Witnessing the limitations of existing jailbreak methods, we introduce Generative Adversarial Suffix Prompter (GASP), a novel framework that combines human-readable prompt generation with Latent Bayesian Optimization (LBO) to improve adversarial suffix creation in a fully black-box setting. GASP leverages LBO to craft adversarial suffixes by efficiently exploring continuous embedding spaces, gradually optimizing the model to improve attack efficacy while balancing prompt coherence through a targeted iterative refinement procedure. Our experiments show that GASP can generate natural jailbreak prompts, significantly improving attack success rates, reducing training times, and accelerating inference speed, thus making it an efficient and scalable solution for red-teaming LLMs.



## **8. RAG-Thief: Scalable Extraction of Private Data from Retrieval-Augmented Generation Applications with Agent-based Attacks**

cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14110v1) [paper-pdf](http://arxiv.org/pdf/2411.14110v1)

**Authors**: Changyue Jiang, Xudong Pan, Geng Hong, Chenfu Bao, Min Yang

**Abstract**: While large language models (LLMs) have achieved notable success in generative tasks, they still face limitations, such as lacking up-to-date knowledge and producing hallucinations. Retrieval-Augmented Generation (RAG) enhances LLM performance by integrating external knowledge bases, providing additional context which significantly improves accuracy and knowledge coverage. However, building these external knowledge bases often requires substantial resources and may involve sensitive information. In this paper, we propose an agent-based automated privacy attack called RAG-Thief, which can extract a scalable amount of private data from the private database used in RAG applications. We conduct a systematic study on the privacy risks associated with RAG applications, revealing that the vulnerability of LLMs makes the private knowledge bases suffer significant privacy risks. Unlike previous manual attacks which rely on traditional prompt injection techniques, RAG-Thief starts with an initial adversarial query and learns from model responses, progressively generating new queries to extract as many chunks from the knowledge base as possible. Experimental results show that our RAG-Thief can extract over 70% information from the private knowledge bases within customized RAG applications deployed on local machines and real-world platforms, including OpenAI's GPTs and ByteDance's Coze. Our findings highlight the privacy vulnerabilities in current RAG applications and underscore the pressing need for stronger safeguards.



## **9. AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer**

cs.CV

32 pages, 12 figures

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2406.08298v5) [paper-pdf](http://arxiv.org/pdf/2406.08298v5)

**Authors**: Yitao Xu, Tong Zhang, Sabine Süsstrunk

**Abstract**: Vision Transformers (ViTs) demonstrate remarkable performance in image classification through visual-token interaction learning, particularly when equipped with local information via region attention or convolutions. Although such architectures improve the feature aggregation from different granularities, they often fail to contribute to the robustness of the networks. Neural Cellular Automata (NCA) enables the modeling of global visual-token representations through local interactions, with its training strategies and architecture design conferring strong generalization ability and robustness against noisy input. In this paper, we propose Adaptor Neural Cellular Automata (AdaNCA) for Vision Transformers that uses NCA as plug-and-play adaptors between ViT layers, thus enhancing ViT's performance and robustness against adversarial samples as well as out-of-distribution inputs. To overcome the large computational overhead of standard NCAs, we propose Dynamic Interaction for more efficient interaction learning. Using our analysis of AdaNCA placement and robustness improvement, we also develop an algorithm for identifying the most effective insertion points for AdaNCA. With less than a 3% increase in parameters, AdaNCA contributes to more than 10% absolute improvement in accuracy under adversarial attacks on the ImageNet1K benchmark. Moreover, we demonstrate with extensive evaluations across eight robustness benchmarks and four ViT architectures that AdaNCA, as a plug-and-play module, consistently improves the robustness of ViTs.



## **10. Verifying the Robustness of Automatic Credibility Assessment**

cs.CL

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2303.08032v3) [paper-pdf](http://arxiv.org/pdf/2303.08032v3)

**Authors**: Piotr Przybyła, Alexander Shvets, Horacio Saggion

**Abstract**: Text classification methods have been widely investigated as a way to detect content of low credibility: fake news, social media bots, propaganda, etc. Quite accurate models (likely based on deep neural networks) help in moderating public electronic platforms and often cause content creators to face rejection of their submissions or removal of already published texts. Having the incentive to evade further detection, content creators try to come up with a slightly modified version of the text (known as an attack with an adversarial example) that exploit the weaknesses of classifiers and result in a different output. Here we systematically test the robustness of common text classifiers against available attacking techniques and discover that, indeed, meaning-preserving changes in input text can mislead the models. The approaches we test focus on finding vulnerable spans in text and replacing individual characters or words, taking into account the similarity between the original and replacement content. We also introduce BODEGA: a benchmark for testing both victim models and attack methods on four misinformation detection tasks in an evaluation framework designed to simulate real use-cases of content moderation. The attacked tasks include (1) fact checking and detection of (2) hyperpartisan news, (3) propaganda and (4) rumours. Our experimental results show that modern large language models are often more vulnerable to attacks than previous, smaller solutions, e.g. attacks on GEMMA being up to 27\% more successful than those on BERT. Finally, we manually analyse a subset adversarial examples and check what kinds of modifications are used in successful attacks.



## **11. Robust Data-Driven Predictive Control for Mixed Platoons under Noise and Attacks**

eess.SY

16 pages, 7 figures

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.13924v1) [paper-pdf](http://arxiv.org/pdf/2411.13924v1)

**Authors**: Shuai Li, Chaoyi Chen, Haotian Zheng, Jiawei Wang, Qing Xu, Jianqiang Wang, Keqiang Li

**Abstract**: Controlling mixed platoons, which consist of both connected and automated vehicles (CAVs) and human-driven vehicles (HDVs), poses significant challenges due to the uncertain and unknown human driving behaviors. Data-driven control methods offer promising solutions by leveraging available trajectory data, but their performance can be compromised by process noise and adversarial attacks. To address this issue, this paper proposes a Robust Data-EnablEd Predictive Leading Cruise Control (RDeeP-LCC) framework based on data-driven reachability analysis. The framework over-approximates system dynamics under noise and attack using a matrix zonotope set derived from data, and develops a stabilizing feedback control law. By decoupling the mixed platoon system into nominal and error components, we employ data-driven reachability sets to recursively compute error reachable sets that account for noise and attacks, and obtain tightened safety constraints of the nominal system. This leads to a robust data-driven predictive control framework, solved in a tube-based control manner. Numerical simulations and human-in-the-loop experiments validate that the RDeeP-LCC method significantly enhances the robustness of mixed platoons, improving mixed traffic stability and safety against practical noise and attacks.



## **12. Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems**

cs.CR

Accepted at NDSS 2025

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2311.00207v3) [paper-pdf](http://arxiv.org/pdf/2311.00207v3)

**Authors**: Jung-Woo Chang, Ke Sun, Nasimeh Heydaribeni, Seira Hidano, Xinyu Zhang, Farinaz Koushanfar

**Abstract**: Machine Learning (ML) has been instrumental in enabling joint transceiver optimization by merging all physical layer blocks of the end-to-end wireless communication systems. Although there have been a number of adversarial attacks on ML-based wireless systems, the existing methods do not provide a comprehensive view including multi-modality of the source data, common physical layer protocols, and wireless domain constraints. This paper proposes Magmaw, a novel wireless attack methodology capable of generating universal adversarial perturbations for any multimodal signal transmitted over a wireless channel. We further introduce new objectives for adversarial attacks on downstream applications. We adopt the widely-used defenses to verify the resilience of Magmaw. For proof-of-concept evaluation, we build a real-time wireless attack platform using a software-defined radio system. Experimental results demonstrate that Magmaw causes significant performance degradation even in the presence of strong defense mechanisms. Furthermore, we validate the performance of Magmaw in two case studies: encrypted communication channel and channel modality-based ML model.



## **13. Towards Understanding Adversarial Transferability in Federated Learning**

cs.LG

Published in Transactions on Machine Learning Research (TMLR)  (11/2024)

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2310.00616v2) [paper-pdf](http://arxiv.org/pdf/2310.00616v2)

**Authors**: Yijiang Li, Ying Gao, Haohan Wang

**Abstract**: We investigate a specific security risk in FL: a group of malicious clients has impacted the model during training by disguising their identities and acting as benign clients but later switching to an adversarial role. They use their data, which was part of the training set, to train a substitute model and conduct transferable adversarial attacks against the federated model. This type of attack is subtle and hard to detect because these clients initially appear to be benign.   The key question we address is: How robust is the FL system to such covert attacks, especially compared to traditional centralized learning systems? We empirically show that the proposed attack imposes a high security risk to current FL systems. By using only 3\% of the client's data, we achieve the highest attack rate of over 80\%. To further offer a full understanding of the challenges the FL system faces in transferable attacks, we provide a comprehensive analysis over the transfer robustness of FL across a spectrum of configurations. Surprisingly, FL systems show a higher level of robustness than their centralized counterparts, especially when both systems are equally good at handling regular, non-malicious data.   We attribute this increased robustness to two main factors: 1) Decentralized Data Training: Each client trains the model on its own data, reducing the overall impact of any single malicious client. 2) Model Update Averaging: The updates from each client are averaged together, further diluting any malicious alterations. Both practical experiments and theoretical analysis support our conclusions. This research not only sheds light on the resilience of FL systems against hidden attacks but also raises important considerations for their future application and development.



## **14. TransLinkGuard: Safeguarding Transformer Models Against Model Stealing in Edge Deployment**

cs.CR

Accepted by ACM MM24 Conference

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2404.11121v2) [paper-pdf](http://arxiv.org/pdf/2404.11121v2)

**Authors**: Qinfeng Li, Zhiqiang Shen, Zhenghan Qin, Yangfan Xie, Xuhong Zhang, Tianyu Du, Jianwei Yin

**Abstract**: Proprietary large language models (LLMs) have been widely applied in various scenarios. Additionally, deploying LLMs on edge devices is trending for efficiency and privacy reasons. However, edge deployment of proprietary LLMs introduces new security challenges: edge-deployed models are exposed as white-box accessible to users, enabling adversaries to conduct effective model stealing (MS) attacks. Unfortunately, existing defense mechanisms fail to provide effective protection. Specifically, we identify four critical protection properties that existing methods fail to simultaneously satisfy: (1) maintaining protection after a model is physically copied; (2) authorizing model access at request level; (3) safeguarding runtime reverse engineering; (4) achieving high security with negligible runtime overhead. To address the above issues, we propose TransLinkGuard, a plug-and-play model protection approach against model stealing on edge devices. The core part of TransLinkGuard is a lightweight authorization module residing in a secure environment, e.g., TEE. The authorization module can freshly authorize each request based on its input. Extensive experiments show that TransLinkGuard achieves the same security protection as the black-box security guarantees with negligible overhead.



## **15. Physical Adversarial Attack meets Computer Vision: A Decade Survey**

cs.CV

Published at IEEE TPAMI. GitHub:https://github.com/weihui1308/PAA

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2209.15179v4) [paper-pdf](http://arxiv.org/pdf/2209.15179v4)

**Authors**: Hui Wei, Hao Tang, Xuemei Jia, Zhixiang Wang, Hanxun Yu, Zhubo Li, Shin'ichi Satoh, Luc Van Gool, Zheng Wang

**Abstract**: Despite the impressive achievements of Deep Neural Networks (DNNs) in computer vision, their vulnerability to adversarial attacks remains a critical concern. Extensive research has demonstrated that incorporating sophisticated perturbations into input images can lead to a catastrophic degradation in DNNs' performance. This perplexing phenomenon not only exists in the digital space but also in the physical world. Consequently, it becomes imperative to evaluate the security of DNNs-based systems to ensure their safe deployment in real-world scenarios, particularly in security-sensitive applications. To facilitate a profound understanding of this topic, this paper presents a comprehensive overview of physical adversarial attacks. Firstly, we distill four general steps for launching physical adversarial attacks. Building upon this foundation, we uncover the pervasive role of artifacts carrying adversarial perturbations in the physical world. These artifacts influence each step. To denote them, we introduce a new term: adversarial medium. Then, we take the first step to systematically evaluate the performance of physical adversarial attacks, taking the adversarial medium as a first attempt. Our proposed evaluation metric, hiPAA, comprises six perspectives: Effectiveness, Stealthiness, Robustness, Practicability, Aesthetics, and Economics. We also provide comparative results across task categories, together with insightful observations and suggestions for future research directions.



## **16. A Survey on Adversarial Robustness of LiDAR-based Machine Learning Perception in Autonomous Vehicles**

cs.LG

20 pages, 2 figures

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.13778v1) [paper-pdf](http://arxiv.org/pdf/2411.13778v1)

**Authors**: Junae Kim, Amardeep Kaur

**Abstract**: In autonomous driving, the combination of AI and vehicular technology offers great potential. However, this amalgamation comes with vulnerabilities to adversarial attacks. This survey focuses on the intersection of Adversarial Machine Learning (AML) and autonomous systems, with a specific focus on LiDAR-based systems. We comprehensively explore the threat landscape, encompassing cyber-attacks on sensors and adversarial perturbations. Additionally, we investigate defensive strategies employed in countering these threats. This paper endeavors to present a concise overview of the challenges and advances in securing autonomous driving systems against adversarial threats, emphasizing the need for robust defenses to ensure safety and security.



## **17. WaterPark: A Robustness Assessment of Language Model Watermarking**

cs.CR

22 pages

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13425v1) [paper-pdf](http://arxiv.org/pdf/2411.13425v1)

**Authors**: Jiacheng Liang, Zian Wang, Lauren Hong, Shouling Ji, Ting Wang

**Abstract**: To mitigate the misuse of large language models (LLMs), such as disinformation, automated phishing, and academic cheating, there is a pressing need for the capability of identifying LLM-generated texts. Watermarking emerges as one promising solution: it plants statistical signals into LLMs' generative processes and subsequently verifies whether LLMs produce given texts. Various watermarking methods (``watermarkers'') have been proposed; yet, due to the lack of unified evaluation platforms, many critical questions remain under-explored: i) What are the strengths/limitations of various watermarkers, especially their attack robustness? ii) How do various design choices impact their robustness? iii) How to optimally operate watermarkers in adversarial environments?   To fill this gap, we systematize existing LLM watermarkers and watermark removal attacks, mapping out their design spaces. We then develop WaterPark, a unified platform that integrates 10 state-of-the-art watermarkers and 12 representative attacks. More importantly, leveraging WaterPark, we conduct a comprehensive assessment of existing watermarkers, unveiling the impact of various design choices on their attack robustness. For instance, a watermarker's resilience to increasingly intensive attacks hinges on its context dependency. We further explore the best practices to operate watermarkers in adversarial environments. For instance, using a generic detector alongside a watermark-specific detector improves the security of vulnerable watermarkers. We believe our study sheds light on current LLM watermarking techniques while WaterPark serves as a valuable testbed to facilitate future research.



## **18. CopyrightMeter: Revisiting Copyright Protection in Text-to-image Models**

cs.CR

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13144v1) [paper-pdf](http://arxiv.org/pdf/2411.13144v1)

**Authors**: Naen Xu, Changjiang Li, Tianyu Du, Minxi Li, Wenjie Luo, Jiacheng Liang, Yuyuan Li, Xuhong Zhang, Meng Han, Jianwei Yin, Ting Wang

**Abstract**: Text-to-image diffusion models have emerged as powerful tools for generating high-quality images from textual descriptions. However, their increasing popularity has raised significant copyright concerns, as these models can be misused to reproduce copyrighted content without authorization. In response, recent studies have proposed various copyright protection methods, including adversarial perturbation, concept erasure, and watermarking techniques. However, their effectiveness and robustness against advanced attacks remain largely unexplored. Moreover, the lack of unified evaluation frameworks has hindered systematic comparison and fair assessment of different approaches. To bridge this gap, we systematize existing copyright protection methods and attacks, providing a unified taxonomy of their design spaces. We then develop CopyrightMeter, a unified evaluation framework that incorporates 17 state-of-the-art protections and 16 representative attacks. Leveraging CopyrightMeter, we comprehensively evaluate protection methods across multiple dimensions, thereby uncovering how different design choices impact fidelity, efficacy, and resilience under attacks. Our analysis reveals several key findings: (i) most protections (16/17) are not resilient against attacks; (ii) the "best" protection varies depending on the target priority; (iii) more advanced attacks significantly promote the upgrading of protections. These insights provide concrete guidance for developing more robust protection methods, while its unified evaluation protocol establishes a standard benchmark for future copyright protection research in text-to-image generation.



## **19. TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models**

cs.CV

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13136v1) [paper-pdf](http://arxiv.org/pdf/2411.13136v1)

**Authors**: Xin Wang, Kai Chen, Jiaming Zhang, Jingjing Chen, Xingjun Ma

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated excellent zero-shot generalizability across various downstream tasks. However, recent studies have shown that the inference performance of CLIP can be greatly degraded by small adversarial perturbations, especially its visual modality, posing significant safety threats. To mitigate this vulnerability, in this paper, we propose a novel defense method called Test-Time Adversarial Prompt Tuning (TAPT) to enhance the inference robustness of CLIP against visual adversarial attacks. TAPT is a test-time defense method that learns defensive bimodal (textual and visual) prompts to robustify the inference process of CLIP. Specifically, it is an unsupervised method that optimizes the defensive prompts for each test sample by minimizing a multi-view entropy and aligning adversarial-clean distributions. We evaluate the effectiveness of TAPT on 11 benchmark datasets, including ImageNet and 10 other zero-shot datasets, demonstrating that it enhances the zero-shot adversarial robustness of the original CLIP by at least 48.9% against AutoAttack (AA), while largely maintaining performance on clean examples. Moreover, TAPT outperforms existing adversarial prompt tuning methods across various backbones, achieving an average robustness improvement of at least 36.6%.



## **20. Disco Intelligent Omni-Surfaces: 360-degree Fully-Passive Jamming Attacks**

eess.SP

This paper has been submitted to IEEE TWC for possible publication

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.12985v1) [paper-pdf](http://arxiv.org/pdf/2411.12985v1)

**Authors**: Huan Huang, Hongliang Zhang, Jide Yuan, Luyao Sun, Yitian Wang, Weidong Mei, Boya Di, Yi Cai, Zhu Han

**Abstract**: Intelligent omni-surfaces (IOSs) with 360-degree electromagnetic radiation significantly improves the performance of wireless systems, while an adversarial IOS also poses a significant potential risk for physical layer security. In this paper, we propose a "DISCO" IOS (DIOS) based fully-passive jammer (FPJ) that can launch omnidirectional fully-passive jamming attacks. In the proposed DIOS-based FPJ, the interrelated refractive and reflective (R&R) coefficients of the adversarial IOS are randomly generated, acting like a "DISCO" that distributes wireless energy radiated by the base station. By introducing active channel aging (ACA) during channel coherence time, the DIOS-based FPJ can perform omnidirectional fully-passive jamming without neither jamming power nor channel knowledge of legitimate users (LUs). To characterize the impact of the DIOS-based PFJ, we derive the statistical characteristics of DIOS-jammed channels based on two widely-used IOS models, i.e., the constant-amplitude model and the variable-amplitude model. Consequently, the asymptotic analysis of the ergodic achievable sum rates under the DIOS-based omnidirectional fully-passive jamming is given based on the derived stochastic characteristics for both the two IOS models. Based on the derived analysis, the omnidirectional jamming impact of the proposed DIOS-based FPJ implemented by a constant-amplitude IOS does not depend on either the quantization number or the stochastic distribution of the DIOS coefficients, while the conclusion does not hold on when a variable-amplitude IOS is used. Numerical results based on one-bit quantization of the IOS phase shifts are provided to verify the effectiveness of the derived theoretical analysis. The proposed DIOS-based FPJ can not only launch omnidirectional fully-passive jamming, but also improve the jamming impact by about 55% at 10 dBm transmit power per LU.



## **21. Efficient Model-Stealing Attacks Against Inductive Graph Neural Networks**

cs.LG

Accepted at ECAI - 27th European Conference on Artificial  Intelligence

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2405.12295v4) [paper-pdf](http://arxiv.org/pdf/2405.12295v4)

**Authors**: Marcin Podhajski, Jan Dubiński, Franziska Boenisch, Adam Dziedzic, Agnieszka Pregowska, Tomasz P. Michalak

**Abstract**: Graph Neural Networks (GNNs) are recognized as potent tools for processing real-world data organized in graph structures. Especially inductive GNNs, which allow for the processing of graph-structured data without relying on predefined graph structures, are becoming increasingly important in a wide range of applications. As such these networks become attractive targets for model-stealing attacks where an adversary seeks to replicate the functionality of the targeted network. Significant efforts have been devoted to developing model-stealing attacks that extract models trained on images and texts. However, little attention has been given to stealing GNNs trained on graph data. This paper identifies a new method of performing unsupervised model-stealing attacks against inductive GNNs, utilizing graph contrastive learning and spectral graph augmentations to efficiently extract information from the targeted model. The new type of attack is thoroughly evaluated on six datasets and the results show that our approach outperforms the current state-of-the-art by Shen et al. (2021). In particular, our attack surpasses the baseline across all benchmarks, attaining superior fidelity and downstream accuracy of the stolen model while necessitating fewer queries directed toward the target model.



## **22. Attribute Inference Attacks for Federated Regression Tasks**

cs.LG

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12697v1) [paper-pdf](http://arxiv.org/pdf/2411.12697v1)

**Authors**: Francesco Diana, Othmane Marfoq, Chuan Xu, Giovanni Neglia, Frédéric Giroire, Eoin Thomas

**Abstract**: Federated Learning (FL) enables multiple clients, such as mobile phones and IoT devices, to collaboratively train a global machine learning model while keeping their data localized. However, recent studies have revealed that the training phase of FL is vulnerable to reconstruction attacks, such as attribute inference attacks (AIA), where adversaries exploit exchanged messages and auxiliary public information to uncover sensitive attributes of targeted clients. While these attacks have been extensively studied in the context of classification tasks, their impact on regression tasks remains largely unexplored. In this paper, we address this gap by proposing novel model-based AIAs specifically designed for regression tasks in FL environments. Our approach considers scenarios where adversaries can either eavesdrop on exchanged messages or directly interfere with the training process. We benchmark our proposed attacks against state-of-the-art methods using real-world datasets. The results demonstrate a significant increase in reconstruction accuracy, particularly in heterogeneous client datasets, a common scenario in FL. The efficacy of our model-based AIAs makes them better candidates for empirically quantifying privacy leakage for federated regression tasks.



## **23. Stochastic BIQA: Median Randomized Smoothing for Certified Blind Image Quality Assessment**

eess.IV

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12575v1) [paper-pdf](http://arxiv.org/pdf/2411.12575v1)

**Authors**: Ekaterina Shumitskaya, Mikhail Pautov, Dmitriy Vatolin, Anastasia Antsiferova

**Abstract**: Most modern No-Reference Image-Quality Assessment (NR-IQA) metrics are based on neural networks vulnerable to adversarial attacks. Attacks on such metrics lead to incorrect image/video quality predictions, which poses significant risks, especially in public benchmarks. Developers of image processing algorithms may unfairly increase the score of a target IQA metric without improving the actual quality of the adversarial image. Although some empirical defenses for IQA metrics were proposed, they do not provide theoretical guarantees and may be vulnerable to adaptive attacks. This work focuses on developing a provably robust no-reference IQA metric. Our method is based on Median Smoothing (MS) combined with an additional convolution denoiser with ranking loss to improve the SROCC and PLCC scores of the defended IQA metric. Compared with two prior methods on three datasets, our method exhibited superior SROCC and PLCC scores while maintaining comparable certified guarantees.



## **24. Variational Bayesian Bow tie Neural Networks with Shrinkage**

stat.ML

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.11132v2) [paper-pdf](http://arxiv.org/pdf/2411.11132v2)

**Authors**: Alisa Sheinkman, Sara Wade

**Abstract**: Despite the dominant role of deep models in machine learning, limitations persist, including overconfident predictions, susceptibility to adversarial attacks, and underestimation of variability in predictions. The Bayesian paradigm provides a natural framework to overcome such issues and has become the gold standard for uncertainty estimation with deep models, also providing improved accuracy and a framework for tuning critical hyperparameters. However, exact Bayesian inference is challenging, typically involving variational algorithms that impose strong independence and distributional assumptions. Moreover, existing methods are sensitive to the architectural choice of the network. We address these issues by constructing a relaxed version of the standard feed-forward rectified neural network, and employing Polya-Gamma data augmentation tricks to render a conditionally linear and Gaussian model. Additionally, we use sparsity-promoting priors on the weights of the neural network for data-driven architectural design. To approximate the posterior, we derive a variational inference algorithm that avoids distributional assumptions and independence across layers and is a faster alternative to the usual Markov Chain Monte Carlo schemes.



## **25. NMT-Obfuscator Attack: Ignore a sentence in translation with only one word**

cs.CL

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12473v1) [paper-pdf](http://arxiv.org/pdf/2411.12473v1)

**Authors**: Sahar Sadrizadeh, César Descalzo, Ljiljana Dolamic, Pascal Frossard

**Abstract**: Neural Machine Translation systems are used in diverse applications due to their impressive performance. However, recent studies have shown that these systems are vulnerable to carefully crafted small perturbations to their inputs, known as adversarial attacks. In this paper, we propose a new type of adversarial attack against NMT models. In this attack, we find a word to be added between two sentences such that the second sentence is ignored and not translated by the NMT model. The word added between the two sentences is such that the whole adversarial text is natural in the source language. This type of attack can be harmful in practical scenarios since the attacker can hide malicious information in the automatic translation made by the target NMT model. Our experiments show that different NMT models and translation tasks are vulnerable to this type of attack. Our attack can successfully force the NMT models to ignore the second part of the input in the translation for more than 50% of all cases while being able to maintain low perplexity for the whole input.



## **26. Efficient Verifiable Differential Privacy with Input Authenticity in the Local and Shuffle Model**

cs.CR

21 pages, 13 figures, 2 tables; accepted for publication in the  Proceedings on the 25th Privacy Enhancing Technologies Symposium (PoPETs)  2025

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2406.18940v2) [paper-pdf](http://arxiv.org/pdf/2406.18940v2)

**Authors**: Tariq Bontekoe, Hassan Jameel Asghar, Fatih Turkmen

**Abstract**: Local differential privacy (LDP) enables the efficient release of aggregate statistics without having to trust the central server (aggregator), as in the central model of differential privacy, and simultaneously protects a client's sensitive data. The shuffle model with LDP provides an additional layer of privacy, by disconnecting the link between clients and the aggregator. However, LDP has been shown to be vulnerable to malicious clients who can perform both input and output manipulation attacks, i.e., before and after applying the LDP mechanism, to skew the aggregator's results. In this work, we show how to prevent malicious clients from compromising LDP schemes. Our only realistic assumption is that the initial raw input is authenticated; the rest of the processing pipeline, e.g., formatting the input and applying the LDP mechanism, may be under adversarial control. We give several real-world examples where this assumption is justified. Our proposed schemes for verifiable LDP (VLDP), prevent both input and output manipulation attacks against generic LDP mechanisms, requiring only one-time interaction between client and server, unlike existing alternatives [37, 43]. Most importantly, we are the first to provide an efficient scheme for VLDP in the shuffle model. We describe, and prove security of, two schemes for VLDP in the local model, and one in the shuffle model. We show that all schemes are highly practical, with client run times of less than 2 seconds, and server run times of 5-7 milliseconds per client.



## **27. DeTrigger: A Gradient-Centric Approach to Backdoor Attack Mitigation in Federated Learning**

cs.LG

14 pages

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12220v1) [paper-pdf](http://arxiv.org/pdf/2411.12220v1)

**Authors**: Kichang Lee, Yujin Shin, Jonghyuk Yun, Jun Han, JeongGil Ko

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed devices while preserving local data privacy, making it ideal for mobile and embedded systems. However, the decentralized nature of FL also opens vulnerabilities to model poisoning attacks, particularly backdoor attacks, where adversaries implant trigger patterns to manipulate model predictions. In this paper, we propose DeTrigger, a scalable and efficient backdoor-robust federated learning framework that leverages insights from adversarial attack methodologies. By employing gradient analysis with temperature scaling, DeTrigger detects and isolates backdoor triggers, allowing for precise model weight pruning of backdoor activations without sacrificing benign model knowledge. Extensive evaluations across four widely used datasets demonstrate that DeTrigger achieves up to 251x faster detection than traditional methods and mitigates backdoor attacks by up to 98.9%, with minimal impact on global model accuracy. Our findings establish DeTrigger as a robust and scalable solution to protect federated learning environments against sophisticated backdoor threats.



## **28. Architectural Patterns for Designing Quantum Artificial Intelligence Systems**

cs.SE

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.10487v2) [paper-pdf](http://arxiv.org/pdf/2411.10487v2)

**Authors**: Mykhailo Klymenko, Thong Hoang, Xiwei Xu, Zhenchang Xing, Muhammad Usman, Qinghua Lu, Liming Zhu

**Abstract**: Utilising quantum computing technology to enhance artificial intelligence systems is expected to improve training and inference times, increase robustness against noise and adversarial attacks, and reduce the number of parameters without compromising accuracy. However, moving beyond proof-of-concept or simulations to develop practical applications of these systems while ensuring high software quality faces significant challenges due to the limitations of quantum hardware and the underdeveloped knowledge base in software engineering for such systems. In this work, we have conducted a systematic mapping study to identify the challenges and solutions associated with the software architecture of quantum-enhanced artificial intelligence systems. Our review uncovered several architectural patterns that describe how quantum components can be integrated into inference engines, as well as middleware patterns that facilitate communication between classical and quantum components. These insights have been compiled into a catalog of architectural patterns. Each pattern realises a trade-off between efficiency, scalability, trainability, simplicity, portability and deployability, and other software quality attributes.



## **29. Adversarial Multi-Agent Reinforcement Learning for Proactive False Data Injection Detection**

eess.SY

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12130v1) [paper-pdf](http://arxiv.org/pdf/2411.12130v1)

**Authors**: Kejun Chen, Truc Nguyen, Malik Hassanaly

**Abstract**: Smart inverters are instrumental in the integration of renewable and distributed energy resources (DERs) into the electric grid. Such inverters rely on communication layers for continuous control and monitoring, potentially exposing them to cyber-physical attacks such as false data injection attacks (FDIAs). We propose to construct a defense strategy against a priori unknown FDIAs with a multi-agent reinforcement learning (MARL) framework. The first agent is an adversary that simulates and discovers various FDIA strategies, while the second agent is a defender in charge of detecting and localizing FDIAs. This approach enables the defender to be trained against new FDIAs continuously generated by the adversary. The numerical results demonstrate that the proposed MARL defender outperforms a supervised offline defender. Additionally, we show that the detection skills of an MARL defender can be combined with that of an offline defender through a transfer learning approach.



## **30. Theoretical Corrections and the Leveraging of Reinforcement Learning to Enhance Triangle Attack**

cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.12071v1) [paper-pdf](http://arxiv.org/pdf/2411.12071v1)

**Authors**: Nicole Meng, Caleb Manicke, David Chen, Yingjie Lao, Caiwen Ding, Pengyu Hong, Kaleel Mahmood

**Abstract**: Adversarial examples represent a serious issue for the application of machine learning models in many sensitive domains. For generating adversarial examples, decision based black-box attacks are one of the most practical techniques as they only require query access to the model. One of the most recently proposed state-of-the-art decision based black-box attacks is Triangle Attack (TA). In this paper, we offer a high-level description of TA and explain potential theoretical limitations. We then propose a new decision based black-box attack, Triangle Attack with Reinforcement Learning (TARL). Our new attack addresses the limits of TA by leveraging reinforcement learning. This creates an attack that can achieve similar, if not better, attack accuracy than TA with half as many queries on state-of-the-art classifiers and defenses across ImageNet and CIFAR-10.



## **31. Exploring adversarial robustness of JPEG AI: methodology, comparison and new methods**

eess.IV

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11795v1) [paper-pdf](http://arxiv.org/pdf/2411.11795v1)

**Authors**: Egor Kovalev, Georgii Bychkov, Khaled Abud, Aleksandr Gushchin, Anna Chistyakova, Sergey Lavrushkin, Dmitriy Vatolin, Anastasia Antsiferova

**Abstract**: Adversarial robustness of neural networks is an increasingly important area of research, combining studies on computer vision models, large language models (LLMs), and others. With the release of JPEG AI - the first standard for end-to-end neural image compression (NIC) methods - the question of its robustness has become critically significant. JPEG AI is among the first international, real-world applications of neural-network-based models to be embedded in consumer devices. However, research on NIC robustness has been limited to open-source codecs and a narrow range of attacks. This paper proposes a new methodology for measuring NIC robustness to adversarial attacks. We present the first large-scale evaluation of JPEG AI's robustness, comparing it with other NIC models. Our evaluation results and code are publicly available online (link is hidden for a blind review).



## **32. Robust Subgraph Learning by Monitoring Early Training Representations**

cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2403.09901v2) [paper-pdf](http://arxiv.org/pdf/2403.09901v2)

**Authors**: Sepideh Neshatfar, Salimeh Yasaei Sekeh

**Abstract**: Graph neural networks (GNNs) have attracted significant attention for their outstanding performance in graph learning and node classification tasks. However, their vulnerability to adversarial attacks, particularly through susceptible nodes, poses a challenge in decision-making. The need for robust graph summarization is evident in adversarial challenges resulting from the propagation of attacks throughout the entire graph. In this paper, we address both performance and adversarial robustness in graph input by introducing the novel technique SHERD (Subgraph Learning Hale through Early Training Representation Distances). SHERD leverages information from layers of a partially trained graph convolutional network (GCN) to detect susceptible nodes during adversarial attacks using standard distance metrics. The method identifies "vulnerable (bad)" nodes and removes such nodes to form a robust subgraph while maintaining node classification performance. Through our experiments, we demonstrate the increased performance of SHERD in enhancing robustness by comparing the network's performance on original and subgraph inputs against various baselines alongside existing adversarial attacks. Our experiments across multiple datasets, including citation datasets such as Cora, Citeseer, and Pubmed, as well as microanatomical tissue structures of cell graphs in the placenta, highlight that SHERD not only achieves substantial improvement in robust performance but also outperforms several baselines in terms of node classification accuracy and computational complexity.



## **33. Eidos: Efficient, Imperceptible Adversarial 3D Point Clouds**

cs.CV

Preprint

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2405.14210v2) [paper-pdf](http://arxiv.org/pdf/2405.14210v2)

**Authors**: Hanwei Zhang, Luo Cheng, Qisong He, Wei Huang, Renjue Li, Ronan Sicre, Xiaowei Huang, Holger Hermanns, Lijun Zhang

**Abstract**: Classification of 3D point clouds is a challenging machine learning (ML) task with important real-world applications in a spectrum from autonomous driving and robot-assisted surgery to earth observation from low orbit. As with other ML tasks, classification models are notoriously brittle in the presence of adversarial attacks. These are rooted in imperceptible changes to inputs with the effect that a seemingly well-trained model ends up misclassifying the input. This paper adds to the understanding of adversarial attacks by presenting Eidos, a framework providing Efficient Imperceptible aDversarial attacks on 3D pOint cloudS. Eidos supports a diverse set of imperceptibility metrics. It employs an iterative, two-step procedure to identify optimal adversarial examples, thereby enabling a runtime-imperceptibility trade-off. We provide empirical evidence relative to several popular 3D point cloud classification models and several established 3D attack methods, showing Eidos' superiority with respect to efficiency as well as imperceptibility.



## **34. Bitcoin Under Volatile Block Rewards: How Mempool Statistics Can Influence Bitcoin Mining**

cs.CR

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11702v1) [paper-pdf](http://arxiv.org/pdf/2411.11702v1)

**Authors**: Roozbeh Sarenche, Alireza Aghabagherloo, Svetla Nikova, Bart Preneel

**Abstract**: As Bitcoin experiences more halving events, the protocol reward converges to zero, making transaction fees the primary source of miner rewards. This shift in Bitcoin's incentivization mechanism, which introduces volatility into block rewards, could lead to the emergence of new security threats or intensify existing ones. Previous security analyses of Bitcoin have either considered a fixed block reward model or a highly simplified volatile model, overlooking the complexities of Bitcoin's mempool behavior.   In this paper, we present a reinforcement learning-based tool designed to analyze mining strategies under a more realistic volatile model. Our tool uses the Asynchronous Advantage Actor-Critic (A3C) algorithm to derive near-optimal mining strategies while interacting with an environment that models the complexity of the Bitcoin mempool. This tool enables the analysis of adversarial mining strategies, such as selfish mining and undercutting, both before and after difficulty adjustments, providing insights into the effects of mining attacks in both the short and long term.   Our analysis reveals that Bitcoin users' trend of offering higher fees to speed up the inclusion of their transactions in the chain can incentivize payoff-maximizing miners to deviate from the honest strategy. In the fixed reward model, a disincentive for the selfish mining attack is the initial loss period of at least two weeks, during which the attack is not profitable. However, our analysis shows that once the protocol reward diminishes to zero in the future, or even currently on days when transaction fees are comparable to the protocol reward, mining pools might be incentivized to abandon honest mining to gain an immediate profit.



## **35. TrojanRobot: Backdoor Attacks Against Robotic Manipulation in the Physical World**

cs.RO

Initial version with preliminary results. We welcome any feedback or  suggestions

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11683v1) [paper-pdf](http://arxiv.org/pdf/2411.11683v1)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Yichen Wang, Wei Wan, Aishan Liu, Leo Yu Zhang

**Abstract**: Robotic manipulation refers to the autonomous handling and interaction of robots with objects using advanced techniques in robotics and artificial intelligence. The advent of powerful tools such as large language models (LLMs) and large vision-language models (LVLMs) has significantly enhanced the capabilities of these robots in environmental perception and decision-making. However, the introduction of these intelligent agents has led to security threats such as jailbreak attacks and adversarial attacks.   In this research, we take a further step by proposing a backdoor attack specifically targeting robotic manipulation and, for the first time, implementing backdoor attack in the physical world. By embedding a backdoor visual language model into the visual perception module within the robotic system, we successfully mislead the robotic arm's operation in the physical world, given the presence of common items as triggers. Experimental evaluations in the physical world demonstrate the effectiveness of the proposed backdoor attack.



## **36. Few-shot Model Extraction Attacks against Sequential Recommender Systems**

cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11677v1) [paper-pdf](http://arxiv.org/pdf/2411.11677v1)

**Authors**: Hui Zhang, Fu Liu

**Abstract**: Among adversarial attacks against sequential recommender systems, model extraction attacks represent a method to attack sequential recommendation models without prior knowledge. Existing research has primarily concentrated on the adversary's execution of black-box attacks through data-free model extraction. However, a significant gap remains in the literature concerning the development of surrogate models by adversaries with access to few-shot raw data (10\% even less). That is, the challenge of how to construct a surrogate model with high functional similarity within the context of few-shot data scenarios remains an issue that requires resolution.This study addresses this gap by introducing a novel few-shot model extraction framework against sequential recommenders, which is designed to construct a superior surrogate model with the utilization of few-shot data. The proposed few-shot model extraction framework is comprised of two components: an autoregressive augmentation generation strategy and a bidirectional repair loss-facilitated model distillation procedure. Specifically, to generate synthetic data that closely approximate the distribution of raw data, autoregressive augmentation generation strategy integrates a probabilistic interaction sampler to extract inherent dependencies and a synthesis determinant signal module to characterize user behavioral patterns. Subsequently, bidirectional repair loss, which target the discrepancies between the recommendation lists, is designed as auxiliary loss to rectify erroneous predictions from surrogate models, transferring knowledge from the victim model to the surrogate model effectively. Experiments on three datasets show that the proposed few-shot model extraction framework yields superior surrogate models.



## **37. Formal Verification of Deep Neural Networks for Object Detection**

cs.CV

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2407.01295v5) [paper-pdf](http://arxiv.org/pdf/2407.01295v5)

**Authors**: Yizhak Y. Elboher, Avraham Raviv, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep neural networks (DNNs) are widely used in real-world applications, yet they remain vulnerable to errors and adversarial attacks. Formal verification offers a systematic approach to identify and mitigate these vulnerabilities, enhancing model robustness and reliability. While most existing verification methods focus on image classification models, this work extends formal verification to the more complex domain of emph{object detection} models. We propose a formulation for verifying the robustness of such models and demonstrate how state-of-the-art verification tools, originally developed for classification, can be adapted for this purpose. Our experiments, conducted on various datasets and networks, highlight the ability of formal verification to uncover vulnerabilities in object detection models, underscoring the need to extend verification efforts to this domain. This work lays the foundation for further research into formal verification across a broader range of computer vision applications.



## **38. The Dark Side of Trust: Authority Citation-Driven Jailbreak Attacks on Large Language Models**

cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11407v1) [paper-pdf](http://arxiv.org/pdf/2411.11407v1)

**Authors**: Xikang Yang, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: The widespread deployment of large language models (LLMs) across various domains has showcased their immense potential while exposing significant safety vulnerabilities. A major concern is ensuring that LLM-generated content aligns with human values. Existing jailbreak techniques reveal how this alignment can be compromised through specific prompts or adversarial suffixes. In this study, we introduce a new threat: LLMs' bias toward authority. While this inherent bias can improve the quality of outputs generated by LLMs, it also introduces a potential vulnerability, increasing the risk of producing harmful content. Notably, the biases in LLMs is the varying levels of trust given to different types of authoritative information in harmful queries. For example, malware development often favors trust GitHub. To better reveal the risks with LLM, we propose DarkCite, an adaptive authority citation matcher and generator designed for a black-box setting. DarkCite matches optimal citation types to specific risk types and generates authoritative citations relevant to harmful instructions, enabling more effective jailbreak attacks on aligned LLMs.Our experiments show that DarkCite achieves a higher attack success rate (e.g., LLama-2 at 76% versus 68%) than previous methods. To counter this risk, we propose an authenticity and harm verification defense strategy, raising the average defense pass rate (DPR) from 11% to 74%. More importantly, the ability to link citations to the content they encompass has become a foundational function in LLMs, amplifying the influence of LLMs' bias toward authority.



## **39. Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks**

cs.CR

v0.2 (evaluated on more agents)

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.20911v2) [paper-pdf](http://arxiv.org/pdf/2410.20911v2)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: Large language models (LLMs) are increasingly being harnessed to automate cyberattacks, making sophisticated exploits more accessible and scalable. In response, we propose a new defense strategy tailored to counter LLM-driven cyberattacks. We introduce Mantis, a defensive framework that exploits LLMs' susceptibility to adversarial inputs to undermine malicious operations. Upon detecting an automated cyberattack, Mantis plants carefully crafted inputs into system responses, leading the attacker's LLM to disrupt their own operations (passive defense) or even compromise the attacker's machine (active defense). By deploying purposefully vulnerable decoy services to attract the attacker and using dynamic prompt injections for the attacker's LLM, Mantis can autonomously hack back the attacker. In our experiments, Mantis consistently achieved over 95% effectiveness against automated LLM-driven attacks. To foster further research and collaboration, Mantis is available as an open-source tool: https://github.com/pasquini-dario/project_mantis



## **40. Adapting to Cyber Threats: A Phishing Evolution Network (PEN) Framework for Phishing Generation and Analyzing Evolution Patterns using Large Language Models**

cs.CR

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11389v1) [paper-pdf](http://arxiv.org/pdf/2411.11389v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Shuo Wang, Hongsheng Hu, Alsharif Abuadbba, Carsten Rudolph

**Abstract**: Phishing remains a pervasive cyber threat, as attackers craft deceptive emails to lure victims into revealing sensitive information. While Artificial Intelligence (AI), particularly deep learning, has become a key component in defending against phishing attacks, these approaches face critical limitations. The scarcity of publicly available, diverse, and updated data, largely due to privacy concerns, constrains their effectiveness. As phishing tactics evolve rapidly, models trained on limited, outdated data struggle to detect new, sophisticated deception strategies, leaving systems vulnerable to an ever-growing array of attacks. Addressing this gap is essential to strengthening defenses in an increasingly hostile cyber landscape. To address this gap, we propose the Phishing Evolution Network (PEN), a framework leveraging large language models (LLMs) and adversarial training mechanisms to continuously generate high quality and realistic diverse phishing samples, and analyze features of LLM-provided phishing to understand evolving phishing patterns. We evaluate the quality and diversity of phishing samples generated by PEN and find that it produces over 80% realistic phishing samples, effectively expanding phishing datasets across seven dominant types. These PEN-generated samples enhance the performance of current phishing detectors, leading to a 40% improvement in detection accuracy. Additionally, the use of PEN significantly boosts model robustness, reducing detectors' sensitivity to perturbations by up to 60%, thereby decreasing attack success rates under adversarial conditions. When we analyze the phishing patterns that are used in LLM-generated phishing, the cognitive complexity and the tone of time limitation are detected with statistically significant differences compared with existing phishing.



## **41. CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization**

cs.CL

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.12768v1) [paper-pdf](http://arxiv.org/pdf/2411.12768v1)

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun

**Abstract**: Recent studies reveal that Large Language Models (LLMs) are susceptible to backdoor attacks, where adversaries embed hidden triggers that manipulate model responses. Existing backdoor defense methods are primarily designed for vision or classification tasks, and are thus ineffective for text generation tasks, leaving LLMs vulnerable. We introduce Internal Consistency Regularization (CROW), a novel defense using consistency regularization finetuning to address layer-wise inconsistencies caused by backdoor triggers. CROW leverages the intuition that clean models exhibit smooth, consistent transitions in hidden representations across layers, whereas backdoored models show noticeable fluctuation when triggered. By enforcing internal consistency through adversarial perturbations and regularization, CROW neutralizes backdoor effects without requiring clean reference models or prior trigger knowledge, relying only on a small set of clean data. This makes it practical for deployment across various LLM architectures. Experimental results demonstrate that CROW consistently achieves a significant reductions in attack success rates across diverse backdoor strategies and tasks, including negative sentiment, targeted refusal, and code injection, on models such as Llama-2 (7B, 13B), CodeLlama (7B, 13B) and Mistral-7B, while preserving the model's generative capabilities.



## **42. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.23091v4) [paper-pdf](http://arxiv.org/pdf/2410.23091v4)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at \href{https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff}{https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff}



## **43. Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics**

cs.RO

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.13587v1) [paper-pdf](http://arxiv.org/pdf/2411.13587v1)

**Authors**: Taowen Wang, Dongfang Liu, James Chenhao Liang, Wenhao Yang, Qifan Wang, Cheng Han, Jiebo Luo, Ruixiang Tang

**Abstract**: Recently in robotics, Vision-Language-Action (VLA) models have emerged as a transformative approach, enabling robots to execute complex tasks by integrating visual and linguistic inputs within an end-to-end learning framework. While VLA models offer significant capabilities, they also introduce new attack surfaces, making them vulnerable to adversarial attacks. With these vulnerabilities largely unexplored, this paper systematically quantifies the robustness of VLA-based robotic systems. Recognizing the unique demands of robotic execution, our attack objectives target the inherent spatial and functional characteristics of robotic systems. In particular, we introduce an untargeted position-aware attack objective that leverages spatial foundations to destabilize robotic actions, and a targeted attack objective that manipulates the robotic trajectory. Additionally, we design an adversarial patch generation approach that places a small, colorful patch within the camera's view, effectively executing the attack in both digital and physical environments. Our evaluation reveals a marked degradation in task success rates, with up to a 100\% reduction across a suite of simulated robotic tasks, highlighting critical security gaps in current VLA architectures. By unveiling these vulnerabilities and proposing actionable evaluation metrics, this work advances both the understanding and enhancement of safety for VLA-based robotic systems, underscoring the necessity for developing robust defense strategies prior to physical-world deployments.



## **44. Countering Backdoor Attacks in Image Recognition: A Survey and Evaluation of Mitigation Strategies**

cs.CR

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.11200v1) [paper-pdf](http://arxiv.org/pdf/2411.11200v1)

**Authors**: Kealan Dunnett, Reza Arablouei, Dimity Miller, Volkan Dedeoglu, Raja Jurdak

**Abstract**: The widespread adoption of deep learning across various industries has introduced substantial challenges, particularly in terms of model explainability and security. The inherent complexity of deep learning models, while contributing to their effectiveness, also renders them susceptible to adversarial attacks. Among these, backdoor attacks are especially concerning, as they involve surreptitiously embedding specific triggers within training data, causing the model to exhibit aberrant behavior when presented with input containing the triggers. Such attacks often exploit vulnerabilities in outsourced processes, compromising model integrity without affecting performance on clean (trigger-free) input data. In this paper, we present a comprehensive review of existing mitigation strategies designed to counter backdoor attacks in image recognition. We provide an in-depth analysis of the theoretical foundations, practical efficacy, and limitations of these approaches. In addition, we conduct an extensive benchmarking of sixteen state-of-the-art approaches against eight distinct backdoor attacks, utilizing three datasets, four model architectures, and three poisoning ratios. Our results, derived from 122,236 individual experiments, indicate that while many approaches provide some level of protection, their performance can vary considerably. Furthermore, when compared to two seminal approaches, most newer approaches do not demonstrate substantial improvements in overall performance or consistency across diverse settings. Drawing from these findings, we propose potential directions for developing more effective and generalizable defensive mechanisms in the future.



## **45. Exploiting the Uncoordinated Privacy Protections of Eye Tracking and VR Motion Data for Unauthorized User Identification**

cs.HC

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.12766v1) [paper-pdf](http://arxiv.org/pdf/2411.12766v1)

**Authors**: Samantha Aziz, Oleg Komogortsev

**Abstract**: Virtual reality (VR) devices use a variety of sensors to capture a rich body of user-generated data, which can be misused by malicious parties to covertly infer information about the user. Privacy-enhancing techniques seek to reduce the amount of personally identifying information in sensor data, but these techniques are typically developed for a subset of data streams that are available on the platform, without consideration for the auxiliary information that may be readily available from other sensors. In this paper, we evaluate whether body motion data can be used to circumvent the privacy protections applied to eye tracking data to enable user identification on a VR platform, and vice versa. We empirically show that eye tracking, headset tracking, and hand tracking data are not only informative for inferring user identity on their own, but contain complementary information that can increase the rate of successful user identification. Most importantly, we demonstrate that applying privacy protections to only a subset of the data available in VR can create an opportunity for an adversary to bypass those privacy protections by using other unprotected data streams that are available on the platform, performing a user identification attack as accurately as though a privacy mechanism was never applied. These results highlight a new privacy consideration at the intersection between eye tracking and VR, and emphasizes the need for privacy-enhancing techniques that address multiple technologies comprehensively.



## **46. Optimal Denial-of-Service Attacks Against Partially-Observable Real-Time Monitoring Systems**

cs.IT

arXiv admin note: text overlap with arXiv:2403.04489

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2409.16794v2) [paper-pdf](http://arxiv.org/pdf/2409.16794v2)

**Authors**: Saad Kriouile, Mohamad Assaad, Amira Alloum, Touraj Soleymani

**Abstract**: In this paper, we investigate the impact of denial-of-service attacks on the status updating of a cyber-physical system with one or more sensors connected to a remote monitor via unreliable channels. We approach the problem from the perspective of an adversary that can strategically jam a subset of the channels. The sources are modeled as Markov chains, and the performance of status updating is measured based on the age of incorrect information at the monitor. Our objective is to derive jamming policies that strike a balance between the degradation of the system's performance and the conservation of the adversary's energy. For a single-source scenario, we formulate the problem as a partially-observable Markov decision process, and rigorously prove that the optimal jamming policy is of a threshold form. We then extend the problem to a multi-source scenario. We formulate this problem as a restless multi-armed bandit, and provide a jamming policy based on the Whittle's index. Our numerical results highlight the performance of our policies compared to baseline policies.



## **47. CLMIA: Membership Inference Attacks via Unsupervised Contrastive Learning**

cs.LG

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.11144v1) [paper-pdf](http://arxiv.org/pdf/2411.11144v1)

**Authors**: Depeng Chen, Xiao Liu, Jie Cui, Hong Zhong

**Abstract**: Since machine learning model is often trained on a limited data set, the model is trained multiple times on the same data sample, which causes the model to memorize most of the training set data. Membership Inference Attacks (MIAs) exploit this feature to determine whether a data sample is used for training a machine learning model. However, in realistic scenarios, it is difficult for the adversary to obtain enough qualified samples that mark accurate identity information, especially since most samples are non-members in real world applications. To address this limitation, in this paper, we propose a new attack method called CLMIA, which uses unsupervised contrastive learning to train an attack model without using extra membership status information. Meanwhile, in CLMIA, we require only a small amount of data with known membership status to fine-tune the attack model. Experimental results demonstrate that CLMIA performs better than existing attack methods for different datasets and model structures, especially with data with less marked identity information. In addition, we experimentally find that the attack performs differently for different proportions of labeled identity information for member and non-member data. More analysis proves that our attack method performs better with less labeled identity information, which applies to more realistic scenarios.



## **48. JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit**

cs.CR

18 pages, 10 figures

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.11114v1) [paper-pdf](http://arxiv.org/pdf/2411.11114v1)

**Authors**: Zeqing He, Zhibo Wang, Zhixuan Chu, Huiyu Xu, Rui Zheng, Kui Ren, Chun Chen

**Abstract**: Despite the outstanding performance of Large language models (LLMs) in diverse tasks, they are vulnerable to jailbreak attacks, wherein adversarial prompts are crafted to bypass their security mechanisms and elicit unexpected responses.Although jailbreak attacks are prevalent, the understanding of their underlying mechanisms remains limited. Recent studies have explain typical jailbreaking behavior (e.g., the degree to which the model refuses to respond) of LLMs by analyzing the representation shifts in their latent space caused by jailbreak prompts or identifying key neurons that contribute to the success of these attacks. However, these studies neither explore diverse jailbreak patterns nor provide a fine-grained explanation from the failure of circuit to the changes of representational, leaving significant gaps in uncovering the jailbreak mechanism. In this paper, we propose JailbreakLens, an interpretation framework that analyzes jailbreak mechanisms from both representation (which reveals how jailbreaks alter the model's harmfulness perception) and circuit perspectives (which uncovers the causes of these deceptions by identifying key circuits contributing to the vulnerability), tracking their evolution throughout the entire response generation process. We then conduct an in-depth evaluation of jailbreak behavior on four mainstream LLMs under seven jailbreak strategies. Our evaluation finds that jailbreak prompts amplify components that reinforce affirmative responses while suppressing those that produce refusal. Although this manipulation shifts model representations toward safe clusters to deceive the LLM, leading it to provide detailed responses instead of refusals, it still produce abnormal activation which can be caught in the circuit analysis.



## **49. Exploring the Adversarial Frontier: Quantifying Robustness via Adversarial Hypervolume**

cs.CR

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2403.05100v2) [paper-pdf](http://arxiv.org/pdf/2403.05100v2)

**Authors**: Ping Guo, Cheng Gong, Xi Lin, Zhiyuan Yang, Qingfu Zhang

**Abstract**: The escalating threat of adversarial attacks on deep learning models, particularly in security-critical fields, has underscored the need for robust deep learning systems. Conventional robustness evaluations have relied on adversarial accuracy, which measures a model's performance under a specific perturbation intensity. However, this singular metric does not fully encapsulate the overall resilience of a model against varying degrees of perturbation. To address this gap, we propose a new metric termed adversarial hypervolume, assessing the robustness of deep learning models comprehensively over a range of perturbation intensities from a multi-objective optimization standpoint. This metric allows for an in-depth comparison of defense mechanisms and recognizes the trivial improvements in robustness afforded by less potent defensive strategies. Additionally, we adopt a novel training algorithm that enhances adversarial robustness uniformly across various perturbation intensities, in contrast to methods narrowly focused on optimizing adversarial accuracy. Our extensive empirical studies validate the effectiveness of the adversarial hypervolume metric, demonstrating its ability to reveal subtle differences in robustness that adversarial accuracy overlooks. This research contributes a new measure of robustness and establishes a standard for assessing and benchmarking the resilience of current and future defensive models against adversarial threats.



## **50. Game-Theoretic Neyman-Pearson Detection to Combat Strategic Evasion**

cs.CR

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2206.05276v3) [paper-pdf](http://arxiv.org/pdf/2206.05276v3)

**Authors**: Yinan Hu, Quanyan Zhu

**Abstract**: The security in networked systems depends greatly on recognizing and identifying adversarial behaviors. Traditional detection methods focus on specific categories of attacks and have become inadequate for increasingly stealthy and deceptive attacks that are designed to bypass detection strategically. This work aims to develop a holistic theory to countermeasure such evasive attacks. We focus on extending a fundamental class of statistical-based detection methods based on Neyman-Pearson's (NP) hypothesis testing formulation. We propose game-theoretic frameworks to capture the conflicting relationship between a strategic evasive attacker and an evasion-aware NP detector. By analyzing both the equilibrium behaviors of the attacker and the NP detector, we characterize their performance using Equilibrium Receiver-Operational-Characteristic (EROC) curves. We show that the evasion-aware NP detectors outperform the passive ones in the way that the former can act strategically against the attacker's behavior and adaptively modify their decision rules based on the received messages. In addition, we extend our framework to a sequential setting where the user sends out identically distributed messages. We corroborate the analytical results with a case study of anomaly detection.



