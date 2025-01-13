# Latest Adversarial Attack Papers
**update at 2025-01-13 10:31:33**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Privacy-Preserving Distributed Defense Framework for DC Microgrids Against Exponentially Unbounded False Data Injection Attacks**

eess.SY

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.00588v2) [paper-pdf](http://arxiv.org/pdf/2501.00588v2)

**Authors**: Yi Zhang, Mohamadamin Rajabinezhad, Yichao Wang, Junbo Zhao, Shan Zuo

**Abstract**: This paper introduces a novel, fully distributed control framework for DC microgrids, enhancing resilience against exponentially unbounded false data injection (EU-FDI) attacks. Our framework features a consensus-based secondary control for each converter, effectively addressing these advanced threats. To further safeguard sensitive operational data, a privacy-preserving mechanism is incorporated into the control design, ensuring that critical information remains secure even under adversarial conditions. Rigorous Lyapunov stability analysis confirms the framework's ability to maintain critical DC microgrid operations like voltage regulation and load sharing under EU-FDI threats. The framework's practicality is validated through hardware-in-the-loop experiments, demonstrating its enhanced resilience and robust privacy protection against the complex challenges posed by quick variant FDI attacks.



## **2. Pixel Is Not A Barrier: An Effective Evasion Attack for Pixel-Domain Diffusion Models**

cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2408.11810v2) [paper-pdf](http://arxiv.org/pdf/2408.11810v2)

**Authors**: Chun-Yen Shih, Li-Xuan Peng, Jia-Wei Liao, Ernie Chu, Cheng-Fu Chou, Jun-Cheng Chen

**Abstract**: Diffusion Models have emerged as powerful generative models for high-quality image synthesis, with many subsequent image editing techniques based on them. However, the ease of text-based image editing introduces significant risks, such as malicious editing for scams or intellectual property infringement. Previous works have attempted to safeguard images from diffusion-based editing by adding imperceptible perturbations. These methods are costly and specifically target prevalent Latent Diffusion Models (LDMs), while Pixel-domain Diffusion Models (PDMs) remain largely unexplored and robust against such attacks. Our work addresses this gap by proposing a novel attack framework, AtkPDM. AtkPDM is mainly composed of a feature representation attacking loss that exploits vulnerabilities in denoising UNets and a latent optimization strategy to enhance the naturalness of adversarial images. Extensive experiments demonstrate the effectiveness of our approach in attacking dominant PDM-based editing methods (e.g., SDEdit) while maintaining reasonable fidelity and robustness against common defense methods. Additionally, our framework is extensible to LDMs, achieving comparable performance to existing approaches.



## **3. Adversarial Detection by Approximation of Ensemble Boundary**

cs.LG

27 pages, 7 figures, 5 tables

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2211.10227v5) [paper-pdf](http://arxiv.org/pdf/2211.10227v5)

**Authors**: T. Windeatt

**Abstract**: Despite being effective in many application areas, Deep Neural Networks (DNNs) are vulnerable to being attacked. In object recognition, the attack takes the form of a small perturbation added to an image, that causes the DNN to misclassify, but to a human appears no different. Adversarial attacks lead to defences that are themselves subject to attack, and the attack/ defence strategies provide important information about the properties of DNNs. In this paper, a novel method of detecting adversarial attacks is proposed for an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The ensemble is combined using Walsh coefficients which are capable of approximating Boolean functions and thereby controlling the decision boundary complexity. The hypothesis in this paper is that decision boundaries with high curvature allow adversarial perturbations to be found, but change the curvature of the decision boundary, which is then approximated in a different way by Walsh coefficients compared to the clean images. Besides controlling boundary complexity, the coefficients also measure the correlation with class labels, which may aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class ensemble decision boundaries could in principle be applied to any application area.



## **4. Beyond Optimal Fault Tolerance**

cs.DC

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.06044v1) [paper-pdf](http://arxiv.org/pdf/2501.06044v1)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal ``recoverable fault-tolerance'' achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic ``recovery procedure'' that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.



## **5. Effective faking of verbal deception detection with target-aligned adversarial attacks**

cs.CL

preprint

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05962v1) [paper-pdf](http://arxiv.org/pdf/2501.05962v1)

**Authors**: Bennett Kleinberg, Riccardo Loconte, Bruno Verschuere

**Abstract**: Background: Deception detection through analysing language is a promising avenue using both human judgments and automated machine learning judgments. For both forms of credibility assessment, automated adversarial attacks that rewrite deceptive statements to appear truthful pose a serious threat. Methods: We used a dataset of 243 truthful and 262 fabricated autobiographical stories in a deception detection task for humans and machine learning models. A large language model was tasked to rewrite deceptive statements so that they appear truthful. In Study 1, humans who made a deception judgment or used the detailedness heuristic and two machine learning models (a fine-tuned language model and a simple n-gram model) judged original or adversarial modifications of deceptive statements. In Study 2, we manipulated the target alignment of the modifications, i.e. tailoring the attack to whether the statements would be assessed by humans or computer models. Results: When adversarial modifications were aligned with their target, human (d=-0.07 and d=-0.04) and machine judgments (51% accuracy) dropped to the chance level. When the attack was not aligned with the target, both human heuristics judgments (d=0.30 and d=0.36) and machine learning predictions (63-78%) were significantly better than chance. Conclusions: Easily accessible language models can effectively help anyone fake deception detection efforts both by humans and machine learning models. Robustness against adversarial modifications for humans and machines depends on that target alignment. We close with suggestions on advancing deception research with adversarial attack designs.



## **6. Towards Backdoor Stealthiness in Model Parameter Space**

cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05928v1) [paper-pdf](http://arxiv.org/pdf/2501.05928v1)

**Authors**: Xiaoyun Xu, Zhuoran Liu, Stefanos Koffas, Stjepan Picek

**Abstract**: Recent research on backdoor stealthiness focuses mainly on indistinguishable triggers in input space and inseparable backdoor representations in feature space, aiming to circumvent backdoor defenses that examine these respective spaces. However, existing backdoor attacks are typically designed to resist a specific type of backdoor defense without considering the diverse range of defense mechanisms. Based on this observation, we pose a natural question: Are current backdoor attacks truly a real-world threat when facing diverse practical defenses?   To answer this question, we examine 12 common backdoor attacks that focus on input-space or feature-space stealthiness and 17 diverse representative defenses. Surprisingly, we reveal a critical blind spot: Backdoor attacks designed to be stealthy in input and feature spaces can be mitigated by examining backdoored models in parameter space. To investigate the underlying causes behind this common vulnerability, we study the characteristics of backdoor attacks in the parameter space. Notably, we find that input- and feature-space attacks introduce prominent backdoor-related neurons in parameter space, which are not thoroughly considered by current backdoor attacks. Taking comprehensive stealthiness into account, we propose a novel supply-chain attack called Grond. Grond limits the parameter changes by a simple yet effective module, Adversarial Backdoor Injection (ABI), which adaptively increases the parameter-space stealthiness during the backdoor injection. Extensive experiments demonstrate that Grond outperforms all 12 backdoor attacks against state-of-the-art (including adaptive) defenses on CIFAR-10, GTSRB, and a subset of ImageNet. In addition, we show that ABI consistently improves the effectiveness of common backdoor attacks.



## **7. Backdoor Attacks against No-Reference Image Quality Assessment Models via a Scalable Trigger**

cs.CV

Accept by AAAI 2025

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2412.07277v2) [paper-pdf](http://arxiv.org/pdf/2412.07277v2)

**Authors**: Yi Yu, Song Xia, Xun Lin, Wenhan Yang, Shijian Lu, Yap-peng Tan, Alex Kot

**Abstract**: No-Reference Image Quality Assessment (NR-IQA), responsible for assessing the quality of a single input image without using any reference, plays a critical role in evaluating and optimizing computer vision systems, e.g., low-light enhancement. Recent research indicates that NR-IQA models are susceptible to adversarial attacks, which can significantly alter predicted scores with visually imperceptible perturbations. Despite revealing vulnerabilities, these attack methods have limitations, including high computational demands, untargeted manipulation, limited practical utility in white-box scenarios, and reduced effectiveness in black-box scenarios. To address these challenges, we shift our focus to another significant threat and present a novel poisoning-based backdoor attack against NR-IQA (BAIQA), allowing the attacker to manipulate the IQA model's output to any desired target value by simply adjusting a scaling coefficient $\alpha$ for the trigger. We propose to inject the trigger in the discrete cosine transform (DCT) domain to improve the local invariance of the trigger for countering trigger diminishment in NR-IQA models due to widely adopted data augmentations. Furthermore, the universal adversarial perturbations (UAP) in the DCT space are designed as the trigger, to increase IQA model susceptibility to manipulation and improve attack effectiveness. In addition to the heuristic method for poison-label BAIQA (P-BAIQA), we explore the design of clean-label BAIQA (C-BAIQA), focusing on $\alpha$ sampling and image data refinement, driven by theoretical insights we reveal. Extensive experiments on diverse datasets and various NR-IQA models demonstrate the effectiveness of our attacks. Code can be found at https://github.com/yuyi-sd/BAIQA.



## **8. Image-based Multimodal Models as Intruders: Transferable Multimodal Attacks on Video-based MLLMs**

cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.01042v2) [paper-pdf](http://arxiv.org/pdf/2501.01042v2)

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Bo Han, Yongjie Yin, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models--a common and practical real world scenario--remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal model (IMM) as a surrogate model to craft adversarial video samples. Multimodal interactions and temporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. In addition, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as surrogate model) achieve competitive performance, with average attack success rates of 55.48% on MSVD-QA and 58.26% on MSRVTT-QA for VideoQA tasks, respectively. Our code will be released upon acceptance.



## **9. ActMiner: Applying Causality Tracking and Increment Aligning for Graph-based Cyber Threat Hunting**

cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05793v1) [paper-pdf](http://arxiv.org/pdf/2501.05793v1)

**Authors**: Mingjun Ma, Tiantian Zhu, Tieming Chen, Shuang Li, Jie Ying, Chunlin Xiong, Mingqi Lv, Yan Chen

**Abstract**: To defend against Advanced Persistent Threats on the endpoint, threat hunting employs security knowledge such as cyber threat intelligence to continuously analyze system audit logs through retrospective scanning, querying, or pattern matching, aiming to uncover attack patterns/graphs that traditional detection methods (e.g., recognition for Point of Interest) fail to capture. However, existing threat hunting systems based on provenance graphs face challenges of high false negatives, high false positives, and low efficiency when confronted with diverse attack tactics and voluminous audit logs. To address these issues, we propose a system called Actminer, which constructs query graphs from descriptive relationships in cyber threat intelligence reports for precise threat hunting (i.e., graph alignment) on provenance graphs. First, we present a heuristic search strategy based on equivalent semantic transfer to reduce false negatives. Second, we establish a filtering mechanism based on causal relationships of attack behaviors to mitigate false positives. Finally, we design a tree structure to incrementally update the alignment results, significantly improving hunting efficiency. Evaluation on the DARPA Engagement dataset demonstrates that compared to the SOTA POIROT, Actminer reduces false positives by 39.1%, eliminates all false negatives, and effectively counters adversarial attacks.



## **10. UV-Attack: Physical-World Adversarial Attacks for Person Detection via Dynamic-NeRF-based UV Mapping**

cs.CV

23 pages, 22 figures, submitted to ICLR2025

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05783v1) [paper-pdf](http://arxiv.org/pdf/2501.05783v1)

**Authors**: Yanjie Li, Wenxuan Zhang, Kaisheng Liang, Bin Xiao

**Abstract**: In recent research, adversarial attacks on person detectors using patches or static 3D model-based texture modifications have struggled with low success rates due to the flexible nature of human movement. Modeling the 3D deformations caused by various actions has been a major challenge. Fortunately, advancements in Neural Radiance Fields (NeRF) for dynamic human modeling offer new possibilities. In this paper, we introduce UV-Attack, a groundbreaking approach that achieves high success rates even with extensive and unseen human actions. We address the challenge above by leveraging dynamic-NeRF-based UV mapping. UV-Attack can generate human images across diverse actions and viewpoints, and even create novel actions by sampling from the SMPL parameter space. While dynamic NeRF models are capable of modeling human bodies, modifying clothing textures is challenging because they are embedded in neural network parameters. To tackle this, UV-Attack generates UV maps instead of RGB images and modifies the texture stacks. This approach enables real-time texture edits and makes the attack more practical. We also propose a novel Expectation over Pose Transformation loss (EoPT) to improve the evasion success rate on unseen poses and views. Our experiments show that UV-Attack achieves a 92.75% attack success rate against the FastRCNN model across varied poses in dynamic video settings, significantly outperforming the state-of-the-art AdvCamou attack, which only had a 28.50% ASR. Moreover, we achieve 49.5% ASR on the latest YOLOv8 detector in black-box settings. This work highlights the potential of dynamic NeRF-based UV mapping for creating more effective adversarial attacks on person detectors, addressing key challenges in modeling human movement and texture modification.



## **11. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2408.09093v2) [paper-pdf](http://arxiv.org/pdf/2408.09093v2)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.



## **12. An Efficiency Firmware Verification Framework for Public Key Infrastructure with Smart Grid and Energy Storage System**

cs.CR

10pages, 5 figures

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05722v1) [paper-pdf](http://arxiv.org/pdf/2501.05722v1)

**Authors**: Jhih-Zen Shih, Cheng-Che Chuang, Hong-Sheng Huang, Hsuan-Tung Chen, Hung-Min Sun

**Abstract**: As a critical component of electrical energy infrastructure, the smart grid system has become indispensable to the energy sector. However, the rapid evolution of smart grids has attracted numerous nation-state actors seeking to disrupt the power infrastructure of adversarial nations. This development underscores the urgent need to establish secure mechanisms for firmware updates, with firmware signing and verification serving as pivotal elements in safeguarding system integrity. In this work, we propose a digital signing and verification framework grounded in Public Key Infrastructure (PKI), specifically tailored for resource-constrained devices such as smart meters. The framework utilizes the Concise Binary Object Representation (CBOR) and Object Signing and Encryption (COSE) formats to achieve efficient da-ta encapsulation and robust security features. Our approach not only en-sures the secure deployment of firmware updates against the convergence of information technology (IT) and operational technology (OT) attacks but also addresses performance bottlenecks stemming from device limitations, thereby enhancing the overall reliability and stability of the smart grid sys-tem.



## **13. Adversarial Robustness for Deep Learning-based Wildfire Prediction Models**

cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2412.20006v2) [paper-pdf](http://arxiv.org/pdf/2412.20006v2)

**Authors**: Ryo Ide, Lei Yang

**Abstract**: Smoke detection using Deep Neural Networks (DNNs) is an effective approach for early wildfire detection. However, because smoke is temporally and spatially anomalous, there are limitations in collecting sufficient training data. This raises overfitting and bias concerns in existing DNN-based wildfire detection models. Thus, we introduce WARP (Wildfire Adversarial Robustness Procedure), the first model-agnostic framework for evaluating the adversarial robustness of DNN-based wildfire detection models. WARP addresses limitations in smoke image diversity using global and local adversarial attack methods. The global attack method uses image-contextualized Gaussian noise, while the local attack method uses patch noise injection, tailored to address critical aspects of wildfire detection. Leveraging WARP's model-agnostic capabilities, we assess the adversarial robustness of real-time Convolutional Neural Networks (CNNs) and Transformers. The analysis revealed valuable insights into the models' limitations. Specifically, the global attack method demonstrates that the Transformer model has more than 70% precision degradation than the CNN against global noise. In contrast, the local attack method shows that both models are susceptible to cloud image injections when detecting smoke-positive instances, suggesting a need for model improvements through data augmentation. WARP's comprehensive robustness analysis contributed to the development of wildfire-specific data augmentation strategies, marking a step toward practicality.



## **14. Enforcing Fundamental Relations via Adversarial Attacks on Input Parameter Correlations**

cs.LG

12 pages, 8 figures (Without appendix)

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05588v1) [paper-pdf](http://arxiv.org/pdf/2501.05588v1)

**Authors**: Timo Saala, Lucie Flek, Alexander Jung, Akbar Karimi, Alexander Schmidt, Matthias Schott, Philipp Soldin, Christopher Wiebusch

**Abstract**: Correlations between input parameters play a crucial role in many scientific classification tasks, since these are often related to fundamental laws of nature. For example, in high energy physics, one of the common deep learning use-cases is the classification of signal and background processes in particle collisions. In many such cases, the fundamental principles of the correlations between observables are often better understood than the actual distributions of the observables themselves. In this work, we present a new adversarial attack algorithm called Random Distribution Shuffle Attack (RDSA), emphasizing the correlations between observables in the network rather than individual feature characteristics. Correct application of the proposed novel attack can result in a significant improvement in classification performance - particularly in the context of data augmentation - when using the generated adversaries within adversarial training. Given that correlations between input features are also crucial in many other disciplines. We demonstrate the RDSA effectiveness on six classification tasks, including two particle collision challenges (using CERN Open Data), hand-written digit recognition (MNIST784), human activity recognition (HAR), weather forecasting (Rain in Australia), and ICU patient mortality (MIMIC-IV), demonstrating a general use case beyond fundamental physics for this new type of adversarial attack algorithms.



## **15. CROPS: Model-Agnostic Training-Free Framework for Safe Image Synthesis with Latent Diffusion Models**

cs.CV

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05359v1) [paper-pdf](http://arxiv.org/pdf/2501.05359v1)

**Authors**: Junha Park, Ian Ryu, Jaehui Hwang, Hyungkeun Park, Jiyoon Kim, Jong-Seok Lee

**Abstract**: With advances in diffusion models, image generation has shown significant performance improvements. This raises concerns about the potential abuse of image generation, such as the creation of explicit or violent images, commonly referred to as Not Safe For Work (NSFW) content. To address this, the Stable Diffusion model includes several safety checkers to censor initial text prompts and final output images generated from the model. However, recent research has shown that these safety checkers have vulnerabilities against adversarial attacks, allowing them to generate NSFW images. In this paper, we find that these adversarial attacks are not robust to small changes in text prompts or input latents. Based on this, we propose CROPS (Circular or RandOm Prompts for Safety), a model-agnostic framework that easily defends against adversarial attacks generating NSFW images without requiring additional training. Moreover, we develop an approach that utilizes one-step diffusion models for efficient NSFW detection (CROPS-1), further reducing computational resources. We demonstrate the superiority of our method in terms of performance and applicability.



## **16. Safeguarding System Prompts for LLMs**

cs.CR

15 pages, 5 figures, 2 tables

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2412.13426v2) [paper-pdf](http://arxiv.org/pdf/2412.13426v2)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: Large language models (LLMs) are increasingly utilized in applications where system prompts, which guide model outputs, play a crucial role. These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a robust defense mechanism designed to safeguard system prompts. PromptKeeper tackles two core challenges: reliably detecting prompt leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon detection, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.



## **17. RAG-WM: An Efficient Black-Box Watermarking Approach for Retrieval-Augmented Generation of Large Language Models**

cs.CR

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05249v1) [paper-pdf](http://arxiv.org/pdf/2501.05249v1)

**Authors**: Peizhuo Lv, Mengjie Sun, Hao Wang, Xiaofeng Wang, Shengzhi Zhang, Yuxuan Chen, Kai Chen, Limin Sun

**Abstract**: In recent years, tremendous success has been witnessed in Retrieval-Augmented Generation (RAG), widely used to enhance Large Language Models (LLMs) in domain-specific, knowledge-intensive, and privacy-sensitive tasks. However, attackers may steal those valuable RAGs and deploy or commercialize them, making it essential to detect Intellectual Property (IP) infringement. Most existing ownership protection solutions, such as watermarks, are designed for relational databases and texts. They cannot be directly applied to RAGs because relational database watermarks require white-box access to detect IP infringement, which is unrealistic for the knowledge base in RAGs. Meanwhile, post-processing by the adversary's deployed LLMs typically destructs text watermark information. To address those problems, we propose a novel black-box "knowledge watermark" approach, named RAG-WM, to detect IP infringement of RAGs. RAG-WM uses a multi-LLM interaction framework, comprising a Watermark Generator, Shadow LLM & RAG, and Watermark Discriminator, to create watermark texts based on watermark entity-relationship tuples and inject them into the target RAG. We evaluate RAG-WM across three domain-specific and two privacy-sensitive tasks on four benchmark LLMs. Experimental results show that RAG-WM effectively detects the stolen RAGs in various deployed LLMs. Furthermore, RAG-WM is robust against paraphrasing, unrelated content removal, knowledge insertion, and knowledge expansion attacks. Lastly, RAG-WM can also evade watermark detection approaches, highlighting its promising application in detecting IP infringement of RAG systems.



## **18. DiffAttack: Diffusion-based Timbre-reserved Adversarial Attack in Speaker Identification**

cs.SD

5 pages,4 figures, accepted by ICASSP 2025

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05127v1) [paper-pdf](http://arxiv.org/pdf/2501.05127v1)

**Authors**: Qing Wang, Jixun Yao, Zhaokai Sun, Pengcheng Guo, Lei Xie, John H. L. Hansen

**Abstract**: Being a form of biometric identification, the security of the speaker identification (SID) system is of utmost importance. To better understand the robustness of SID systems, we aim to perform more realistic attacks in SID, which are challenging for both humans and machines to detect. In this study, we propose DiffAttack, a novel timbre-reserved adversarial attack approach that exploits the capability of a diffusion-based voice conversion (DiffVC) model to generate adversarial fake audio with distinct target speaker attribution. By introducing adversarial constraints into the generative process of the diffusion-based voice conversion model, we craft fake samples that effectively mislead target models while preserving speaker-wise characteristics. Specifically, inspired by the use of randomly sampled Gaussian noise in conventional adversarial attacks and diffusion processes, we incorporate adversarial constraints into the reverse diffusion process. These constraints subtly guide the reverse diffusion process toward aligning with the target speaker distribution. Our experiments on the LibriTTS dataset indicate that DiffAttack significantly improves the attack success rate compared to vanilla DiffVC and other methods. Moreover, objective and subjective evaluations demonstrate that introducing adversarial constraints does not compromise the speech quality generated by the DiffVC model.



## **19. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

cs.CL

Our code is publicly available at  https://github.com/UKPLab/POATE-attack

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.01872v2) [paper-pdf](http://arxiv.org/pdf/2501.01872v2)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.



## **20. On Measuring Unnoticeability of Graph Adversarial Attacks: Observations, New Measure, and Applications**

cs.LG

KDD 2025

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05015v1) [paper-pdf](http://arxiv.org/pdf/2501.05015v1)

**Authors**: Hyeonsoo Jo, Hyunjin Hwang, Fanchen Bu, Soo Yong Lee, Chanyoung Park, Kijung Shin

**Abstract**: Adversarial attacks are allegedly unnoticeable. Prior studies have designed attack noticeability measures on graphs, primarily using statistical tests to compare the topology of original and (possibly) attacked graphs. However, we observe two critical limitations in the existing measures. First, because the measures rely on simple rules, attackers can readily enhance their attacks to bypass them, reducing their attack "noticeability" and, yet, maintaining their attack performance. Second, because the measures naively leverage global statistics, such as degree distributions, they may entirely overlook attacks until severe perturbations occur, letting the attacks be almost "totally unnoticeable." To address the limitations, we introduce HideNSeek, a learnable measure for graph attack noticeability. First, to mitigate the bypass problem, HideNSeek learns to distinguish the original and (potential) attack edges using a learnable edge scorer (LEO), which scores each edge on its likelihood of being an attack. Second, to mitigate the overlooking problem, HideNSeek conducts imbalance-aware aggregation of all the edge scores to obtain the final noticeability score. Using six real-world graphs, we empirically demonstrate that HideNSeek effectively alleviates the observed limitations, and LEO (i.e., our learnable edge scorer) outperforms eleven competitors in distinguishing attack edges under five different attack methods. For an additional application, we show that LEO boost the performance of robust GNNs by removing attack-like edges.



## **21. SpaLLM-Guard: Pairing SMS Spam Detection Using Open-source and Commercial LLMs**

cs.CR

17 pages

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.04985v1) [paper-pdf](http://arxiv.org/pdf/2501.04985v1)

**Authors**: Muhammad Salman, Muhammad Ikram, Nardine Basta, Mohamed Ali Kaafar

**Abstract**: The increasing threat of SMS spam, driven by evolving adversarial techniques and concept drift, calls for more robust and adaptive detection methods. In this paper, we evaluate the potential of large language models (LLMs), both open-source and commercial, for SMS spam detection, comparing their performance across zero-shot, few-shot, fine-tuning, and chain-of-thought prompting approaches. Using a comprehensive dataset of SMS messages, we assess the spam detection capabilities of prominent LLMs such as GPT-4, DeepSeek, LLAMA-2, and Mixtral. Our findings reveal that while zero-shot learning provides convenience, it is unreliable for effective spam detection. Few-shot learning, particularly with carefully selected examples, improves detection but exhibits variability across models. Fine-tuning emerges as the most effective strategy, with Mixtral achieving 98.6% accuracy and a balanced false positive and false negative rate below 2%, meeting the criteria for robust spam detection. Furthermore, we explore the resilience of these models to adversarial attacks, finding that fine-tuning significantly enhances robustness against both perceptible and imperceptible manipulations. Lastly, we investigate the impact of concept drift and demonstrate that fine-tuned LLMs, especially when combined with few-shot learning, can mitigate its effects, maintaining high performance even on evolving spam datasets. This study highlights the importance of fine-tuning and tailored learning strategies to deploy LLMs effectively for real-world SMS spam detection



## **22. LayerMix: Enhanced Data Augmentation through Fractal Integration for Robust Deep Learning**

cs.CV

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04861v1) [paper-pdf](http://arxiv.org/pdf/2501.04861v1)

**Authors**: Hafiz Mughees Ahmad, Dario Morle, Afshin Rahimi

**Abstract**: Deep learning models have demonstrated remarkable performance across various computer vision tasks, yet their vulnerability to distribution shifts remains a critical challenge. Despite sophisticated neural network architectures, existing models often struggle to maintain consistent performance when confronted with Out-of-Distribution (OOD) samples, including natural corruptions, adversarial perturbations, and anomalous patterns. We introduce LayerMix, an innovative data augmentation approach that systematically enhances model robustness through structured fractal-based image synthesis. By meticulously integrating structural complexity into training datasets, our method generates semantically consistent synthetic samples that significantly improve neural network generalization capabilities. Unlike traditional augmentation techniques that rely on random transformations, LayerMix employs a structured mixing pipeline that preserves original image semantics while introducing controlled variability. Extensive experiments across multiple benchmark datasets, including CIFAR-10, CIFAR-100, ImageNet-200, and ImageNet-1K demonstrate LayerMixs superior performance in classification accuracy and substantially enhances critical Machine Learning (ML) safety metrics, including resilience to natural image corruptions, robustness against adversarial attacks, improved model calibration and enhanced prediction consistency. LayerMix represents a significant advancement toward developing more reliable and adaptable artificial intelligence systems by addressing the fundamental challenges of deep learning generalization. The code is available at https://github.com/ahmadmughees/layermix.



## **23. Reproducing HotFlip for Corpus Poisoning Attacks in Dense Retrieval**

cs.IR

This paper has been accepted for oral presentation in the  reproducibility track at ECIR 2025

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04802v1) [paper-pdf](http://arxiv.org/pdf/2501.04802v1)

**Authors**: Yongkang Li, Panagiotis Eustratiadis, Evangelos Kanoulas

**Abstract**: HotFlip is a topical gradient-based word substitution method for attacking language models. Recently, this method has been further applied to attack retrieval systems by generating malicious passages that are injected into a corpus, i.e., corpus poisoning. However, HotFlip is known to be computationally inefficient, with the majority of time being spent on gradient accumulation for each query-passage pair during the adversarial token generation phase, making it impossible to generate an adequate number of adversarial passages in a reasonable amount of time. Moreover, the attack method itself assumes access to a set of user queries, a strong assumption that does not correspond to how real-world adversarial attacks are usually performed. In this paper, we first significantly boost the efficiency of HotFlip, reducing the adversarial generation process from 4 hours per document to only 15 minutes, using the same hardware. We further contribute experiments and analysis on two additional tasks: (1) transfer-based black-box attacks, and (2) query-agnostic attacks. Whenever possible, we provide comparisons between the original method and our improved version. Our experiments demonstrate that HotFlip can effectively attack a variety of dense retrievers, with an observed trend that its attack performance diminishes against more advanced and recent methods. Interestingly, we observe that while HotFlip performs poorly in a black-box setting, indicating limited capacity for generalization, in query-agnostic scenarios its performance is correlated to the volume of injected adversarial passages.



## **24. Correlated Privacy Mechanisms for Differentially Private Distributed Mean Estimation**

cs.IT

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2407.03289v2) [paper-pdf](http://arxiv.org/pdf/2407.03289v2)

**Authors**: Sajani Vithana, Viveck R. Cadambe, Flavio P. Calmon, Haewon Jeong

**Abstract**: Differentially private distributed mean estimation (DP-DME) is a fundamental building block in privacy-preserving federated learning, where a central server estimates the mean of $d$-dimensional vectors held by $n$ users while ensuring $(\epsilon,\delta)$-DP. Local differential privacy (LDP) and distributed DP with secure aggregation (SA) are the most common notions of DP used in DP-DME settings with an untrusted server. LDP provides strong resilience to dropouts, colluding users, and adversarial attacks, but suffers from poor utility. In contrast, SA-based DP-DME achieves an $O(n)$ utility gain over LDP in DME, but requires increased communication and computation overheads and complex multi-round protocols to handle dropouts and attacks. In this work, we present a generalized framework for DP-DME, that captures LDP and SA-based mechanisms as extreme cases. Our framework provides a foundation for developing and analyzing a variety of DP-DME protocols that leverage correlated privacy mechanisms across users. To this end, we propose CorDP-DME, a novel DP-DME mechanism based on the correlated Gaussian mechanism, that spans the gap between DME with LDP and distributed DP. We prove that CorDP-DME offers a favorable balance between utility and resilience to dropout and collusion. We provide an information-theoretic analysis of CorDP-DME, and derive theoretical guarantees for utility under any given privacy parameters and dropout/colluding user thresholds. Our results demonstrate that (anti) correlated Gaussian DP mechanisms can significantly improve utility in mean estimation tasks compared to LDP -- even in adversarial settings -- while maintaining better resilience to dropouts and attacks compared to distributed DP.



## **25. Resilient Peer-to-peer Learning based on Adaptive Aggregation**

cs.LG

11 pages

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04610v1) [paper-pdf](http://arxiv.org/pdf/2501.04610v1)

**Authors**: Chandreyee Bhowmick, Xenofon Koutsoukos

**Abstract**: Collaborative learning in peer-to-peer networks offers the benefits of distributed learning while mitigating the risks associated with single points of failure inherent in centralized servers. However, adversarial workers pose potential threats by attempting to inject malicious information into the network. Thus, ensuring the resilience of peer-to-peer learning emerges as a pivotal research objective. The challenge is exacerbated in the presence of non-convex loss functions and non-iid data distributions. This paper introduces a resilient aggregation technique tailored for such scenarios, aimed at fostering similarity among peers' learning processes. The aggregation weights are determined through an optimization procedure, and use the loss function computed using the neighbor's models and individual private data, thereby addressing concerns regarding data privacy in distributed machine learning. Theoretical analysis demonstrates convergence of parameters with non-convex loss functions and non-iid data distributions. Empirical evaluations across three distinct machine learning tasks support the claims. The empirical findings, which encompass a range of diverse attack models, also demonstrate improved accuracy when compared to existing methodologies.



## **26. Tougher Text, Smarter Models: Raising the Bar for Adversarial Defence Benchmarks**

cs.CL

Will be presented as an oral in-person presentation at the conference  of COLING 2025

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.02654v2) [paper-pdf](http://arxiv.org/pdf/2501.02654v2)

**Authors**: Yang Wang, Chenghua Lin

**Abstract**: Recent advancements in natural language processing have highlighted the vulnerability of deep learning models to adversarial attacks. While various defence mechanisms have been proposed, there is a lack of comprehensive benchmarks that evaluate these defences across diverse datasets, models, and tasks. In this work, we address this gap by presenting an extensive benchmark for textual adversarial defence that significantly expands upon previous work. Our benchmark incorporates a wide range of datasets, evaluates state-of-the-art defence mechanisms, and extends the assessment to include critical tasks such as single-sentence classification, similarity and paraphrase identification, natural language inference, and commonsense reasoning. This work not only serves as a valuable resource for researchers and practitioners in the field of adversarial robustness but also identifies key areas for future research in textual adversarial defence. By establishing a new standard for benchmarking in this domain, we aim to accelerate progress towards more robust and reliable natural language processing systems.



## **27. Towards Fair Class-wise Robustness: Class Optimal Distribution Adversarial Training**

cs.LG

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04527v1) [paper-pdf](http://arxiv.org/pdf/2501.04527v1)

**Authors**: Hongxin Zhi, Hongtao Yu, Shaome Li, Xiuming Zhao, Yiteng Wu

**Abstract**: Adversarial training has proven to be a highly effective method for improving the robustness of deep neural networks against adversarial attacks. Nonetheless, it has been observed to exhibit a limitation in terms of robust fairness, characterized by a significant disparity in robustness across different classes. Recent efforts to mitigate this problem have turned to class-wise reweighted methods. However, these methods suffer from a lack of rigorous theoretical analysis and are limited in their exploration of the weight space, as they mainly rely on existing heuristic algorithms or intuition to compute weights. In addition, these methods fail to guarantee the consistency of the optimization direction due to the decoupled optimization of weights and the model parameters. They potentially lead to suboptimal weight assignments and consequently, a suboptimal model. To address these problems, this paper proposes a novel min-max training framework, Class Optimal Distribution Adversarial Training (CODAT), which employs distributionally robust optimization to fully explore the class-wise weight space, thus enabling the identification of the optimal weight with theoretical guarantees. Furthermore, we derive a closed-form optimal solution to the internal maximization and then get a deterministic equivalent objective function, which provides a theoretical basis for the joint optimization of weights and model parameters. Meanwhile, we propose a fairness elasticity coefficient for the evaluation of the algorithm with regard to both robustness and robust fairness. Experimental results on various datasets show that the proposed method can effectively improve the robust fairness of the model and outperform the state-of-the-art approaches.



## **28. Multichannel Steganography: A Provably Secure Hybrid Steganographic Model for Secure Communication**

cs.CR

18 pages, 8 figures, 3 algorithms, This version is a preprint  uploaded to arXiv

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04511v1) [paper-pdf](http://arxiv.org/pdf/2501.04511v1)

**Authors**: Obinna Omego, Michal Bosy

**Abstract**: This study introduces a novel steganographic model that synthesizes Steganography by Cover Modification (CMO) and Steganography by Cover Synthesis (CSY), enhancing both security and undetectability by generating cover messages or parameters while retaining the original cover's form, thus minimizing detection risks and overcoming the limitations of single-method techniques. Building upon this model, a refined Steganographic Communication Protocol is proposed, enhancing resilience against sophisticated threats such as Multichannel Replay Attacks and Multichannel Man-in-the-Middle Attacks, fortifying the protocol against potential tampering and improving upon prior works. To evaluate the security of the proposed protocol, a novel adversarial model is developed simulating a probabilistic polynomial time (PPT) adversary capable of intercepting communications across multiple channels. This model assesses the adversary's ability to compromise the protocol, providing a comprehensive security analysis. Finally, this study explores the practicality and adaptability of the model to both constrained environments like SMS banking and resource-rich settings such as blockchain transactions, demonstrating their potential to enhance financial services and security. These contributions present a robust and adaptable framework for secure steganographic communication, offering practical solutions for secure communications across diverse environments.



## **29. Rethinking Byzantine Robustness in Federated Recommendation from Sparse Aggregation Perspective**

cs.CR

accepted by AAAI 2025

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.03301v2) [paper-pdf](http://arxiv.org/pdf/2501.03301v2)

**Authors**: Zhongjian Zhang, Mengmei Zhang, Xiao Wang, Lingjuan Lyu, Bo Yan, Junping Du, Chuan Shi

**Abstract**: To preserve user privacy in recommender systems, federated recommendation (FR) based on federated learning (FL) emerges, keeping the personal data on the local client and updating a model collaboratively. Unlike FL, FR has a unique sparse aggregation mechanism, where the embedding of each item is updated by only partial clients, instead of full clients in a dense aggregation of general FL. Recently, as an essential principle of FL, model security has received increasing attention, especially for Byzantine attacks, where malicious clients can send arbitrary updates. The problem of exploring the Byzantine robustness of FR is particularly critical since in the domains applying FR, e.g., e-commerce, malicious clients can be injected easily by registering new accounts. However, existing Byzantine works neglect the unique sparse aggregation of FR, making them unsuitable for our problem. Thus, we make the first effort to investigate Byzantine attacks on FR from the perspective of sparse aggregation, which is non-trivial: it is not clear how to define Byzantine robustness under sparse aggregations and design Byzantine attacks under limited knowledge/capability. In this paper, we reformulate the Byzantine robustness under sparse aggregation by defining the aggregation for a single item as the smallest execution unit. Then we propose a family of effective attack strategies, named Spattack, which exploit the vulnerability in sparse aggregation and are categorized along the adversary's knowledge and capability. Extensive experimental results demonstrate that Spattack can effectively prevent convergence and even break down defenses under a few malicious clients, raising alarms for securing FR systems.



## **30. Rethinking Adversarial Attacks in Reinforcement Learning from Policy Distribution Perspective**

cs.LG

10 pages, 2 figures, 2 tables

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.03562v2) [paper-pdf](http://arxiv.org/pdf/2501.03562v2)

**Authors**: Tianyang Duan, Zongyuan Zhang, Zheng Lin, Yue Gao, Ling Xiong, Yong Cui, Hongbin Liang, Xianhao Chen, Heming Cui, Dong Huang

**Abstract**: Deep Reinforcement Learning (DRL) suffers from uncertainties and inaccuracies in the observation signal in realworld applications. Adversarial attack is an effective method for evaluating the robustness of DRL agents. However, existing attack methods targeting individual sampled actions have limited impacts on the overall policy distribution, particularly in continuous action spaces. To address these limitations, we propose the Distribution-Aware Projected Gradient Descent attack (DAPGD). DAPGD uses distribution similarity as the gradient perturbation input to attack the policy network, which leverages the entire policy distribution rather than relying on individual samples. We utilize the Bhattacharyya distance in DAPGD to measure policy similarity, enabling sensitive detection of subtle but critical differences between probability distributions. Our experiment results demonstrate that DAPGD achieves SOTA results compared to the baselines in three robot navigation tasks, achieving an average 22.03% higher reward drop compared to the best baseline.



## **31. Location Privacy Threats and Protections in 6G Vehicular Networks: A Comprehensive Review**

cs.CR

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2305.04503v2) [paper-pdf](http://arxiv.org/pdf/2305.04503v2)

**Authors**: Baihe Ma, Xu Wang, Xiaojie Lin, Yanna Jiang, Caijun Sun, Zhe Wang, Guangsheng Yu, Suirui Zhu, Ying He, Wei Ni, Ren Ping Liu

**Abstract**: Location privacy is critical in vehicular networks, where drivers' trajectories and personal information can be exposed, allowing adversaries to launch data and physical attacks that threaten drivers' safety and personal security. This survey reviews comprehensively different localization techniques, including widely used ones like sensing infrastructure-based, optical vision-based, and cellular radio-based localization, and identifies inadequately addressed location privacy concerns. We classify Location Privacy Preserving Mechanisms (LPPMs) into user-side, server-side, and user-server-interface-based, and evaluate their effectiveness. Our analysis shows that the user-server-interface-based LPPMs have received insufficient attention in the literature, despite their paramount importance in vehicular networks. Further, we examine methods for balancing data utility and privacy protection for existing LPPMs in vehicular networks and highlight emerging challenges from future upper-layer location privacy attacks, wireless technologies, and network convergences. By providing insights into the relationship between localization techniques and location privacy, and evaluating the effectiveness of different LPPMs, this survey can help inform the development of future LPPMs in vehicular networks.



## **32. Proof-of-Learning with Incentive Security**

cs.CR

20 pages, 4 figures

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2404.09005v7) [paper-pdf](http://arxiv.org/pdf/2404.09005v7)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Xi Chen, Hongxu Su, Haibo Xiao, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks, and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.



## **33. Light-weight Fine-tuning Method for Defending Adversarial Noise in Pre-trained Medical Vision-Language Models**

cs.CV

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2407.02716v2) [paper-pdf](http://arxiv.org/pdf/2407.02716v2)

**Authors**: Xu Han, Linghao Jin, Xuezhe Ma, Xiaofeng Liu

**Abstract**: Fine-tuning pre-trained Vision-Language Models (VLMs) has shown remarkable capabilities in medical image and textual depiction synergy. Nevertheless, many pre-training datasets are restricted by patient privacy concerns, potentially containing noise that can adversely affect downstream performance. Moreover, the growing reliance on multi-modal generation exacerbates this issue because of its susceptibility to adversarial attacks. To investigate how VLMs trained on adversarial noisy data perform on downstream medical tasks, we first craft noisy upstream datasets using multi-modal adversarial attacks. Through our comprehensive analysis, we unveil that moderate noise enhances model robustness and transferability, but increasing noise levels negatively impact downstream task performance. To mitigate this issue, we propose rectify adversarial noise (RAN) framework, a recipe designed to effectively defend adversarial attacks and rectify the influence of upstream noise during fine-tuning.



## **34. Synthetic Data Privacy Metrics**

cs.LG

14 pages, 2 figures

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.03941v1) [paper-pdf](http://arxiv.org/pdf/2501.03941v1)

**Authors**: Amy Steier, Lipika Ramaswamy, Andre Manoel, Alexa Haushalter

**Abstract**: Recent advancements in generative AI have made it possible to create synthetic datasets that can be as accurate as real-world data for training AI models, powering statistical insights, and fostering collaboration with sensitive datasets while offering strong privacy guarantees. Effectively measuring the empirical privacy of synthetic data is an important step in the process. However, while there is a multitude of new privacy metrics being published every day, there currently is no standardization. In this paper, we review the pros and cons of popular metrics that include simulations of adversarial attacks. We also review current best practices for amending generative models to enhance the privacy of the data they create (e.g. differential privacy).



## **35. Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection**

cs.CL

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.03940v1) [paper-pdf](http://arxiv.org/pdf/2501.03940v1)

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho

**Abstract**: The rapid advancement in large language models (LLMs) has significantly enhanced their ability to generate coherent and contextually relevant text, raising concerns about the misuse of AI-generated content and making it critical to detect it. However, the task remains challenging, particularly in unseen domains or with unfamiliar LLMs. Leveraging LLM next-token distribution outputs offers a theoretically appealing approach for detection, as they encapsulate insights from the models' extensive pre-training on diverse corpora. Despite its promise, zero-shot methods that attempt to operationalize these outputs have met with limited success. We hypothesize that one of the problems is that they use the mean to aggregate next-token distribution metrics across tokens, when some tokens are naturally easier or harder to predict and should be weighted differently. Based on this idea, we propose the Perplexity Attention Weighted Network (PAWN), which uses the last hidden states of the LLM and positions to weight the sum of a series of features based on metrics from the next-token distribution across the sequence length. Although not zero-shot, our method allows us to cache the last hidden states and next-token distribution metrics on disk, greatly reducing the training resource requirements. PAWN shows competitive and even better performance in-distribution than the strongest baselines (fine-tuned LMs) with a fraction of their trainable parameters. Our model also generalizes better to unseen domains and source models, with smaller variability in the decision boundary across distribution shifts. It is also more robust to adversarial attacks, and if the backbone has multilingual capabilities, it presents decent generalization to languages not seen during supervised training, with LLaMA3-1B reaching a mean macro-averaged F1 score of 81.46% in cross-validation with nine languages.



## **36. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2410.23091v6) [paper-pdf](http://arxiv.org/pdf/2410.23091v6)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff.



## **37. A Volumetric Approach to Privacy of Dynamical Systems**

eess.SY

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.02893v2) [paper-pdf](http://arxiv.org/pdf/2501.02893v2)

**Authors**: Chuanghong Weng, Ehsan Nekouei

**Abstract**: Information-theoretic metrics, such as mutual information, have been widely used to evaluate privacy leakage in dynamic systems. However, these approaches are typically limited to stochastic systems and face computational challenges. In this paper, we introduce a novel volumetric framework for analyzing privacy in systems affected by unknown but bounded noise. Our model considers a dynamic system comprising public and private states, where an observation set of the public state is released. An adversary utilizes the observed public state to infer an uncertainty set of the private state, referred to as the inference attack. We define the evolution dynamics of these inference attacks and quantify the privacy level of the private state using the volume of its uncertainty sets. For linear scalar systems, we derive an explicit formulation of the uncertainty set. For multi-dimensional linear systems, we develop an approximate computation method leveraging interval analysis. We investigate the properties of the proposed volumetric privacy measure and demonstrate that it is bounded by the information gain derived from the observation set. Furthermore, we propose an optimization approach to designing privacy filter using randomization and linear programming based on the proposed privacy measure. The effectiveness of the optimal privacy filter design is evaluated through a production-inventory case study, illustrating its robustness against the inference attack.



## **38. Echomix: a Strong Anonymity System with Messaging**

cs.CR

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.02933v2) [paper-pdf](http://arxiv.org/pdf/2501.02933v2)

**Authors**: Ewa J Infeld, David Stainton, Leif Ryge, Threebit Hacker

**Abstract**: Echomix is a practical mix network framework and a suite of associated protocols providing strong metadata privacy against realistic modern adversaries. It is distinguished from other anonymity systems by a resistance to traffic analysis by global adversaries, compromised contacts and network infrastructure, quantum decryption algorithms, and statistical and confirmation attacks typical for multi-client messaging setting. It is implemented as Katzenpost, a robust software project, and used in multiple deployed systems, and features relatively low latency and bandwidth overhead.   The contributions of this paper are: (1) Improvements on leading mix network designs, supported by rigorous analysis. These include solutions to crucial vulnerabilities to traffic analysis, malicious servers and active attacks. (2) A cryptographic group messaging protocol with strong metadata protection guarantees and reliability. (3) Hybrid post-quantum nested packet encryption.



## **39. Graph Neural Backdoor: Fundamentals, Methodologies, Applications, and Future Directions**

cs.LG

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2406.10573v2) [paper-pdf](http://arxiv.org/pdf/2406.10573v2)

**Authors**: Xiao Yang, Gaolei Li, Jianhua Li

**Abstract**: Graph Neural Networks (GNNs) have significantly advanced various downstream graph-relevant tasks, encompassing recommender systems, molecular structure prediction, social media analysis, etc. Despite the boosts of GNN, recent research has empirically demonstrated its potential vulnerability to backdoor attacks, wherein adversaries employ triggers to poison input samples, inducing GNN to adversary-premeditated malicious outputs. This is typically due to the controlled training process, or the deployment of untrusted models, such as delegating model training to third-party service, leveraging external training sets, and employing pre-trained models from online sources. Although there's an ongoing increase in research on GNN backdoors, comprehensive investigation into this field is lacking. To bridge this gap, we propose the first survey dedicated to GNN backdoors. We begin by outlining the fundamental definition of GNN, followed by the detailed summarization and categorization of current GNN backdoor attacks and defenses based on their technical characteristics and application scenarios. Subsequently, the analysis of the applicability and use cases of GNN backdoors is undertaken. Finally, the exploration of potential research directions of GNN backdoors is presented. This survey aims to explore the principles of graph backdoors, provide insights to defenders, and promote future security research.



## **40. Unraveling Responsiveness of Chained BFT Consensus with Network Delay**

cs.DC

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.03695v1) [paper-pdf](http://arxiv.org/pdf/2501.03695v1)

**Authors**: Yining Tang, Qihang Luo, Runchao Han, Jianyu Niu, Chen Feng, Yinqian Zhang

**Abstract**: With the advancement of blockchain technology, chained Byzantine Fault Tolerant (BFT) protocols have been increasingly adopted in practical systems, making their performance a crucial aspect of the study. In this paper, we introduce a unified framework utilizing Markov Decision Processes (MDP) to model and assess the performance of three prominent chained BFT protocols. Our framework effectively captures complex adversarial behaviors, focusing on two key performance metrics: chain growth and commitment rate. We implement the optimal attack strategies obtained from MDP analysis on an existing evaluation platform for chained BFT protocols and conduct extensive experiments under various settings to validate our theoretical results. Through rigorous theoretical analysis and thorough practical experiments, we provide an in-depth evaluation of chained BFT protocols under diverse attack scenarios, uncovering optimal attack strategies. Contrary to conventional belief, our findings reveal that while responsiveness can enhance performance, it is not universally beneficial across all scenarios. This work not only deepens our understanding of chained BFT protocols, but also offers valuable insights and analytical tools that can inform the design of more robust and efficient protocols.



## **41. Transferable Adversarial Examples with Bayes Approach**

cs.LG

Accepted in AsiaCCS'25

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2208.06538v2) [paper-pdf](http://arxiv.org/pdf/2208.06538v2)

**Authors**: Mingyuan Fan, Cen Chen, Wenmeng Zhou, Yinggui Wang

**Abstract**: The vulnerability of deep neural networks (DNNs) to black-box adversarial attacks is one of the most heated topics in trustworthy AI. In such attacks, the attackers operate without any insider knowledge of the model, making the cross-model transferability of adversarial examples critical. Despite the potential for adversarial examples to be effective across various models, it has been observed that adversarial examples that are specifically crafted for a specific model often exhibit poor transferability. In this paper, we explore the transferability of adversarial examples via the lens of Bayesian approach. Specifically, we leverage Bayesian approach to probe the transferability and then study what constitutes a transferability-promoting prior. Following this, we design two concrete transferability-promoting priors, along with an adaptive dynamic weighting strategy for instances sampled from these priors. Employing these techniques, we present BayAtk. Extensive experiments illustrate the significant effectiveness of BayAtk in crafting more transferable adversarial examples against both undefended and defended black-box models compared to existing state-of-the-art attacks.



## **42. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

cs.CR

Accepted at AAAI 2025

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2408.10738v2) [paper-pdf](http://arxiv.org/pdf/2408.10738v2)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also face notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the relevant top k items from offline knowledge bases, using available information from a webpage, including logos and HTML. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.



## **43. ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates**

cs.CR

This paper is accepted to AAAI 2025

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2406.12935v2) [paper-pdf](http://arxiv.org/pdf/2406.12935v2)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Bill Yuchen Lin, Radha Poovendran

**Abstract**: Large language models (LLMs) are expected to follow instructions from users and engage in conversations. Techniques to enhance LLMs' instruction-following capabilities typically fine-tune them using data structured according to a predefined chat template. Although chat templates are shown to be effective in optimizing LLM performance, their impact on safety alignment of LLMs has been less understood, which is crucial for deploying LLMs safely at scale.   In this paper, we investigate how chat templates affect safety alignment of LLMs. We identify a common vulnerability, named ChatBug, that is introduced by chat templates. Our key insight to identify ChatBug is that the chat templates provide a rigid format that need to be followed by LLMs, but not by users. Hence, a malicious user may not necessarily follow the chat template when prompting LLMs. Instead, malicious users could leverage their knowledge of the chat template and accordingly craft their prompts to bypass safety alignments of LLMs. We develop two attacks to exploit the ChatBug vulnerability. We demonstrate that a malicious user can exploit the ChatBug vulnerability of eight state-of-the-art (SOTA) LLMs and effectively elicit unintended responses from these models. Moreover, we show that ChatBug can be exploited by existing jailbreak attacks to enhance their attack success rates. We investigate potential countermeasures to ChatBug. Our results show that while adversarial training effectively mitigates the ChatBug vulnerability, the victim model incurs significant performance degradation. These results highlight the trade-off between safety alignment and helpfulness. Developing new methods for instruction tuning to balance this trade-off is an open and critical direction for future research



## **44. Countering Backdoor Attacks in Image Recognition: A Survey and Evaluation of Mitigation Strategies**

cs.CR

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2411.11200v2) [paper-pdf](http://arxiv.org/pdf/2411.11200v2)

**Authors**: Kealan Dunnett, Reza Arablouei, Dimity Miller, Volkan Dedeoglu, Raja Jurdak

**Abstract**: The widespread adoption of deep learning across various industries has introduced substantial challenges, particularly in terms of model explainability and security. The inherent complexity of deep learning models, while contributing to their effectiveness, also renders them susceptible to adversarial attacks. Among these, backdoor attacks are especially concerning, as they involve surreptitiously embedding specific triggers within training data, causing the model to exhibit aberrant behavior when presented with input containing the triggers. Such attacks often exploit vulnerabilities in outsourced processes, compromising model integrity without affecting performance on clean (trigger-free) input data. In this paper, we present a comprehensive review of existing mitigation strategies designed to counter backdoor attacks in image recognition. We provide an in-depth analysis of the theoretical foundations, practical efficacy, and limitations of these approaches. In addition, we conduct an extensive benchmarking of sixteen state-of-the-art approaches against eight distinct backdoor attacks, utilizing three datasets, four model architectures, and three poisoning ratios. Our results, derived from 122,236 individual experiments, indicate that while many approaches provide some level of protection, their performance can vary considerably. Furthermore, when compared to two seminal approaches, most newer approaches do not demonstrate substantial improvements in overall performance or consistency across diverse settings. Drawing from these findings, we propose potential directions for developing more effective and generalizable defensive mechanisms in the future.



## **45. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

cs.LG

11 pages, 5 figures

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2412.08099v2) [paper-pdf](http://arxiv.org/pdf/2412.08099v2)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in the field of time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like TimeGPT and LLM-Time with GPT-3.5, GPT-4, LLaMa, and Mistral, show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications.



## **46. When Should Selfish Miners Double-Spend?**

cs.CR

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.03227v1) [paper-pdf](http://arxiv.org/pdf/2501.03227v1)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: Although, both double-spending and selfish-mining attacks have been extensively studied since the ``Bitcoin'' whitepaper of Nakamoto and the ``majority is not enough'' paper of Eyal and Sirer, there has been no rigorous stochastic analysis of an attack that combines the two, except for the complicated MDP models. In this paper, we first combine stubborn and selfish mining attacks, i.e., construct a strategy where the attacker acts stubborn until its private branch reaches a certain length and then switches to act selfish. We provide the optimal stubbornness for each parameter regime. Next, we provide the maximum stubbornness that is still more profitable than honest mining and argue a connection between the level of stubbornness and the $k$-confirmation rule. We show that, at each attack cycle, if the level of stubbornness is higher than $k$, there is a risk of double-spending which comes at no-cost to the adversary. The result can be seen as a guide for picking $k$ in the $k$-confirmation rule in a blockchain design. At each cycle, for a given stubbornness level, we rigorously formulate how great the risk of double-spending is. We provide the minimum double-spend value needed for an attack to be profitable in the regimes where the scheme is less profitable than honest mining. We further modify the attack in the stubborn regime in order to conceal the attack and increase the double-spending probability. Finally, we evaluate the results and provide the optimal and the maximum stubbornness levels for each parameter regime as well as the revenue. As a case study, with Bitcoin's $k=6$ block confirmation rule, we evaluate the revenue and double-spending risk of the attacks for each pool parameter.



## **47. The Robustness of Spiking Neural Networks in Federated Learning with Compression Against Non-omniscient Byzantine Attacks**

cs.CR

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.03306v1) [paper-pdf](http://arxiv.org/pdf/2501.03306v1)

**Authors**: Manh V. Nguyen, Liang Zhao, Bobin Deng, Shaoen Wu

**Abstract**: Spiking Neural Networks (SNNs), which offer exceptional energy efficiency for inference, and Federated Learning (FL), which offers privacy-preserving distributed training, is a rising area of interest that highly beneficial towards Internet of Things (IoT) devices. Despite this, research that tackles Byzantine attacks and bandwidth limitation in FL-SNNs, both poses significant threats on model convergence and training times, still remains largely unexplored. Going beyond proposing a solution for both of these problems, in this work we highlight the dual benefits of FL-SNNs, against non-omniscient Byzantine adversaries (ones that restrict attackers access to local clients datasets), and greater communication efficiency, over FL-ANNs. Specifically, we discovered that a simple integration of Top-\k{appa} sparsification into the FL apparatus can help leverage the advantages of the SNN models in both greatly reducing bandwidth usage and significantly boosting the robustness of FL training against non-omniscient Byzantine adversaries. Most notably, we saw a massive improvement of roughly 40% accuracy gain in FL-SNNs training under the lethal MinMax attack



## **48. Leader Rotation Is Not Enough: Scrutinizing Leadership Democracy of Chained BFT Consensus**

cs.CR

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02970v1) [paper-pdf](http://arxiv.org/pdf/2501.02970v1)

**Authors**: Yining Tang, Runchao Han, Jianyu Niu, Chen Feng, Yinqian Zhang

**Abstract**: With the growing popularity of blockchains, modern chained BFT protocols combining chaining and leader rotation to obtain better efficiency and leadership democracy have received increasing interest. Although the efficiency provisions of chained BFT protocols have been thoroughly analyzed, the leadership democracy has received little attention in prior work. In this paper, we scrutinize the leadership democracy of four representative chained BFT protocols, especially under attack. To this end, we propose a unified framework with two evaluation metrics, i.e., chain quality and censorship resilience, and quantitatively analyze chosen protocols through the Markov Decision Process (MDP). With this framework, we further examine the impact of two key components, i.e., voting pattern and leader rotation on leadership democracy. Our results indicate that leader rotation is not enough to provide the leadership democracy guarantee; an adversary could utilize the design, e.g., voting pattern, to deteriorate the leadership democracy significantly. Based on the analysis results, we propose customized countermeasures for three evaluated protocols to improve their leadership democracy with only slight protocol overhead and no change of consensus rules. We also discuss future directions toward building more democratic chained BFT protocols.



## **49. Seeing the Whole in the Parts in Self-Supervised Representation Learning**

cs.LG

20 pages

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02860v1) [paper-pdf](http://arxiv.org/pdf/2501.02860v1)

**Authors**: Arthur Aubret, Céline Teulière, Jochen Triesch

**Abstract**: Recent successes in self-supervised learning (SSL) model spatial co-occurrences of visual features either by masking portions of an image or by aggressively cropping it. Here, we propose a new way to model spatial co-occurrences by aligning local representations (before pooling) with a global image representation. We present CO-SSL, a family of instance discrimination methods and show that it outperforms previous methods on several datasets, including ImageNet-1K where it achieves 71.5% of Top-1 accuracy with 100 pre-training epochs. CO-SSL is also more robust to noise corruption, internal corruption, small adversarial attacks, and large training crop sizes. Our analysis further indicates that CO-SSL learns highly redundant local representations, which offers an explanation for its robustness. Overall, our work suggests that aligning local and global representations may be a powerful principle of unsupervised category learning.



## **50. MBTSAD: Mitigating Backdoors in Language Models Based on Token Splitting and Attention Distillation**

cs.CR

Accepted by ICTAI 2024

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02754v1) [paper-pdf](http://arxiv.org/pdf/2501.02754v1)

**Authors**: Yidong Ding, Jiafei Niu, Ping Yi

**Abstract**: In recent years, attention-based models have excelled across various domains but remain vulnerable to backdoor attacks, often from downloading or fine-tuning on poisoned datasets. Many current methods to mitigate backdoors in NLP models rely on the pre-trained (unfine-tuned) weights, but these methods fail in scenarios where the pre-trained weights are not available. In this work, we propose MBTSAD, which can mitigate backdoors in the language model by utilizing only a small subset of clean data and does not require pre-trained weights. Specifically, MBTSAD retrains the backdoored model on a dataset generated by token splitting. Then MBTSAD leverages attention distillation, the retrained model is the teacher model, and the original backdoored model is the student model. Experimental results demonstrate that MBTSAD achieves comparable backdoor mitigation performance as the methods based on pre-trained weights while maintaining the performance on clean data. MBTSAD does not rely on pre-trained weights, enhancing its utility in scenarios where pre-trained weights are inaccessible. In addition, we simplify the min-max problem of adversarial training and visualize text representations to discover that the token splitting method in MBTSAD's first step generates Out-of-Distribution (OOD) data, leading the model to learn more generalized features and eliminate backdoor patterns.



