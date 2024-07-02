# Latest Adversarial Attack Papers
**update at 2024-07-02 15:09:58**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. On the Abuse and Detection of Polyglot Files**

cs.CR

18 pages, 11 figures

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01529v1) [paper-pdf](http://arxiv.org/pdf/2407.01529v1)

**Authors**: Luke Koch, Sean Oesch, Amul Chaulagain, Jared Dixon, Matthew Dixon, Mike Huettal, Amir Sadovnik, Cory Watson, Brian Weber, Jacob Hartman, Richard Patulski

**Abstract**: A polyglot is a file that is valid in two or more formats. Polyglot files pose a problem for malware detection systems that route files to format-specific detectors/signatures, as well as file upload and sanitization tools. In this work we found that existing file-format and embedded-file detection tools, even those developed specifically for polyglot files, fail to reliably detect polyglot files used in the wild, leaving organizations vulnerable to attack. To address this issue, we studied the use of polyglot files by malicious actors in the wild, finding $30$ polyglot samples and $15$ attack chains that leveraged polyglot files. In this report, we highlight two well-known APTs whose cyber attack chains relied on polyglot files to bypass detection mechanisms. Using knowledge from our survey of polyglot usage in the wild -- the first of its kind -- we created a novel data set based on adversary techniques. We then trained a machine learning detection solution, PolyConv, using this data set. PolyConv achieves a precision-recall area-under-curve score of $0.999$ with an F1 score of $99.20$% for polyglot detection and $99.47$% for file-format identification, significantly outperforming all other tools tested. We developed a content disarmament and reconstruction tool, ImSan, that successfully sanitized $100$% of the tested image-based polyglots, which were the most common type found via the survey. Our work provides concrete tools and suggestions to enable defenders to better defend themselves against polyglot files, as well as directions for future work to create more robust file specifications and methods of disarmament.



## **2. Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement**

cs.CL

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01461v1) [paper-pdf](http://arxiv.org/pdf/2407.01461v1)

**Authors**: Zisu Huang, Xiaohua Wang, Feiran Zhang, Zhibo Xu, Cenyuan Zhang, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .



## **3. Cutting through buggy adversarial example defenses: fixing 1 line of code breaks Sabre**

cs.CR

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2405.03672v3) [paper-pdf](http://arxiv.org/pdf/2405.03672v3)

**Authors**: Nicholas Carlini

**Abstract**: Sabre is a defense to adversarial examples that was accepted at IEEE S&P 2024. We first reveal significant flaws in the evaluation that point to clear signs of gradient masking. We then show the cause of this gradient masking: a bug in the original evaluation code. By fixing a single line of code in the original repository, we reduce Sabre's robust accuracy to 0%. In response to this, the authors modify the defense and introduce a new defense component not described in the original paper. But this fix contains a second bug; modifying one more line of code reduces robust accuracy to below baseline levels. After we released the first version of our paper online, the authors introduced another change to the defense; by commenting out one line of code during attack we reduce the robust accuracy to 0% again.



## **4. Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt**

cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2406.04031v2) [paper-pdf](http://arxiv.org/pdf/2406.04031v2)

**Authors**: Zonghao Ying, Aishan Liu, Tianyuan Zhang, Zhengmin Yu, Siyuan Liang, Xianglong Liu, Dacheng Tao

**Abstract**: In the realm of large vision language models (LVLMs), jailbreak attacks serve as a red-teaming approach to bypass guardrails and uncover safety implications. Existing jailbreaks predominantly focus on the visual modality, perturbing solely visual inputs in the prompt for attacks. However, they fall short when confronted with aligned models that fuse visual and textual features simultaneously for generation. To address this limitation, this paper introduces the Bi-Modal Adversarial Prompt Attack (BAP), which executes jailbreaks by optimizing textual and visual prompts cohesively. Initially, we adversarially embed universally harmful perturbations in an image, guided by a few-shot query-agnostic corpus (e.g., affirmative prefixes and negative inhibitions). This process ensures that image prompt LVLMs to respond positively to any harmful queries. Subsequently, leveraging the adversarial image, we optimize textual prompts with specific harmful intent. In particular, we utilize a large language model to analyze jailbreak failures and employ chain-of-thought reasoning to refine textual prompts through a feedback-iteration manner. To validate the efficacy of our approach, we conducted extensive evaluations on various datasets and LVLMs, demonstrating that our method significantly outperforms other methods by large margins (+29.03% in attack success rate on average). Additionally, we showcase the potential of our attacks on black-box commercial LVLMs, such as Gemini and ChatGLM.



## **5. Formal Verification of Object Detection**

cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01295v1) [paper-pdf](http://arxiv.org/pdf/2407.01295v1)

**Authors**: Avraham Raviv, Yizhak Y. Elboher, Michelle Aluf-Medina, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep Neural Networks (DNNs) are ubiquitous in real-world applications, yet they remain vulnerable to errors and adversarial attacks. This work tackles the challenge of applying formal verification to ensure the safety of computer vision models, extending verification beyond image classification to object detection. We propose a general formulation for certifying the robustness of object detection models using formal verification and outline implementation strategies compatible with state-of-the-art verification tools. Our approach enables the application of these tools, originally designed for verifying classification models, to object detection. We define various attacks for object detection, illustrating the diverse ways adversarial inputs can compromise neural network outputs. Our experiments, conducted on several common datasets and networks, reveal potential errors in object detection models, highlighting system vulnerabilities and emphasizing the need for expanding formal verification to these new domains. This work paves the way for further research in integrating formal verification across a broader range of computer vision applications.



## **6. DeepiSign-G: Generic Watermark to Stamp Hidden DNN Parameters for Self-contained Tracking**

cs.CR

13 pages

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01260v1) [paper-pdf](http://arxiv.org/pdf/2407.01260v1)

**Authors**: Alsharif Abuadbba, Nicholas Rhodes, Kristen Moore, Bushra Sabir, Shuo Wang, Yansong Gao

**Abstract**: Deep learning solutions in critical domains like autonomous vehicles, facial recognition, and sentiment analysis require caution due to the severe consequences of errors. Research shows these models are vulnerable to adversarial attacks, such as data poisoning and neural trojaning, which can covertly manipulate model behavior, compromising reliability and safety. Current defense strategies like watermarking have limitations: they fail to detect all model modifications and primarily focus on attacks on CNNs in the image domain, neglecting other critical architectures like RNNs.   To address these gaps, we introduce DeepiSign-G, a versatile watermarking approach designed for comprehensive verification of leading DNN architectures, including CNNs and RNNs. DeepiSign-G enhances model security by embedding an invisible watermark within the Walsh-Hadamard transform coefficients of the model's parameters. This watermark is highly sensitive and fragile, ensuring prompt detection of any modifications. Unlike traditional hashing techniques, DeepiSign-G allows substantial metadata incorporation directly within the model, enabling detailed, self-contained tracking and verification.   We demonstrate DeepiSign-G's applicability across various architectures, including CNN models (VGG, ResNets, DenseNet) and RNNs (Text sentiment classifier). We experiment with four popular datasets: VGG Face, CIFAR10, GTSRB Traffic Sign, and Large Movie Review. We also evaluate DeepiSign-G under five potential attacks. Our comprehensive evaluation confirms that DeepiSign-G effectively detects these attacks without compromising CNN and RNN model performance, highlighting its efficacy as a robust security measure for deep learning applications. Detection of integrity breaches is nearly perfect, while hiding only a bit in approximately 1% of the Walsh-Hadamard coefficients.



## **7. QUEEN: Query Unlearning against Model Extraction**

cs.CR

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01251v1) [paper-pdf](http://arxiv.org/pdf/2407.01251v1)

**Authors**: Huajie Chen, Tianqing Zhu, Lefeng Zhang, Bo Liu, Derui Wang, Wanlei Zhou, Minhui Xue

**Abstract**: Model extraction attacks currently pose a non-negligible threat to the security and privacy of deep learning models. By querying the model with a small dataset and usingthe query results as the ground-truth labels, an adversary can steal a piracy model with performance comparable to the original model. Two key issues that cause the threat are, on the one hand, accurate and unlimited queries can be obtained by the adversary; on the other hand, the adversary can aggregate the query results to train the model step by step. The existing defenses usually employ model watermarking or fingerprinting to protect the ownership. However, these methods cannot proactively prevent the violation from happening. To mitigate the threat, we propose QUEEN (QUEry unlEarNing) that proactively launches counterattacks on potential model extraction attacks from the very beginning. To limit the potential threat, QUEEN has sensitivity measurement and outputs perturbation that prevents the adversary from training a piracy model with high performance. In sensitivity measurement, QUEEN measures the single query sensitivity by its distance from the center of its cluster in the feature space. To reduce the learning accuracy of attacks, for the highly sensitive query batch, QUEEN applies query unlearning, which is implemented by gradient reverse to perturb the softmax output such that the piracy model will generate reverse gradients to worsen its performance unconsciously. Experiments show that QUEEN outperforms the state-of-the-art defenses against various model extraction attacks with a relatively low cost to the model accuracy. The artifact is publicly available at https://anonymous.4open.science/r/queen implementation-5408/.



## **8. Multi-View Black-Box Physical Attacks on Infrared Pedestrian Detectors Using Adversarial Infrared Grid**

cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01168v1) [paper-pdf](http://arxiv.org/pdf/2407.01168v1)

**Authors**: Kalibinuer Tiliwalidi, Chengyin Hu, Weiwen Shi

**Abstract**: While extensive research exists on physical adversarial attacks within the visible spectrum, studies on such techniques in the infrared spectrum are limited. Infrared object detectors are vital in modern technological applications but are susceptible to adversarial attacks, posing significant security threats. Previous studies using physical perturbations like light bulb arrays and aerogels for white-box attacks, or hot and cold patches for black-box attacks, have proven impractical or limited in multi-view support. To address these issues, we propose the Adversarial Infrared Grid (AdvGrid), which models perturbations in a grid format and uses a genetic algorithm for black-box optimization. These perturbations are cyclically applied to various parts of a pedestrian's clothing to facilitate multi-view black-box physical attacks on infrared pedestrian detectors. Extensive experiments validate AdvGrid's effectiveness, stealthiness, and robustness. The method achieves attack success rates of 80.00\% in digital environments and 91.86\% in physical environments, outperforming baseline methods. Additionally, the average attack success rate exceeds 50\% against mainstream detectors, demonstrating AdvGrid's robustness. Our analyses include ablation studies, transfer attacks, and adversarial defenses, confirming the method's superiority.



## **9. Unaligning Everything: Or Aligning Any Text to Any Image in Multimodal Models**

cs.CV

14 pages, 14 figures

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01157v1) [paper-pdf](http://arxiv.org/pdf/2407.01157v1)

**Authors**: Shaeke Salman, Md Montasir Bin Shams, Xiuwen Liu

**Abstract**: Utilizing a shared embedding space, emerging multimodal models exhibit unprecedented zero-shot capabilities. However, the shared embedding space could lead to new vulnerabilities if different modalities can be misaligned. In this paper, we extend and utilize a recently developed effective gradient-based procedure that allows us to match the embedding of a given text by minimally modifying an image. Using the procedure, we show that we can align the embeddings of distinguishable texts to any image through unnoticeable adversarial attacks in joint image-text models, revealing that semantically unrelated images can have embeddings of identical texts and at the same time visually indistinguishable images can be matched to the embeddings of very different texts. Our technique achieves 100\% success rate when it is applied to text datasets and images from multiple sources. Without overcoming the vulnerability, multimodal models cannot robustly align inputs from different modalities in a semantically meaningful way. \textbf{Warning: the text data used in this paper are toxic in nature and may be offensive to some readers.}



## **10. SecGenAI: Enhancing Security of Cloud-based Generative AI Applications within Australian Critical Technologies of National Interest**

cs.CR

10 pages, 4 figures, 9 tables, submitted to the 2024 11th  International Conference on Soft Computing & Machine Intelligence (ISCMI  2024)

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01110v1) [paper-pdf](http://arxiv.org/pdf/2407.01110v1)

**Authors**: Christoforus Yoga Haryanto, Minh Hieu Vu, Trung Duc Nguyen, Emily Lomempow, Yulia Nurliana, Sona Taheri

**Abstract**: The rapid advancement of Generative AI (GenAI) technologies offers transformative opportunities within Australia's critical technologies of national interest while introducing unique security challenges. This paper presents SecGenAI, a comprehensive security framework for cloud-based GenAI applications, with a focus on Retrieval-Augmented Generation (RAG) systems. SecGenAI addresses functional, infrastructure, and governance requirements, integrating end-to-end security analysis to generate specifications emphasizing data privacy, secure deployment, and shared responsibility models. Aligned with Australian Privacy Principles, AI Ethics Principles, and guidelines from the Australian Cyber Security Centre and Digital Transformation Agency, SecGenAI mitigates threats such as data leakage, adversarial attacks, and model inversion. The framework's novel approach combines advanced machine learning techniques with robust security measures, ensuring compliance with Australian regulations while enhancing the reliability and trustworthiness of GenAI systems. This research contributes to the field of intelligent systems by providing actionable strategies for secure GenAI implementation in industry, fostering innovation in AI applications, and safeguarding national interests.



## **11. DifAttack++: Query-Efficient Black-Box Adversarial Attack via Hierarchical Disentangled Feature Space in Cross-Domain**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2309.14585 An  extension of the AAAI24 paper "DifAttack: Query-Efficient Black-Box Attack  via Disentangled Feature Space."

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2406.03017v3) [paper-pdf](http://arxiv.org/pdf/2406.03017v3)

**Authors**: Jun Liu, Jiantao Zhou, Jiandian Zeng, Jinyu Tian, Zheng Li

**Abstract**: This work investigates efficient score-based black-box adversarial attacks with a high Attack Success Rate (\textbf{ASR}) and good generalizability. We design a novel attack method based on a hierarchical DIsentangled Feature space, called \textbf{DifAttack++}, which differs significantly from the existing ones operating over the entire feature space. Specifically, DifAttack++ firstly disentangles an image's latent feature into an Adversarial Feature (\textbf{AF}) and a Visual Feature (\textbf{VF}) via an autoencoder equipped with our specially designed Hierarchical Decouple-Fusion (\textbf{HDF}) module, where the AF dominates the adversarial capability of an image, while the VF largely determines its visual appearance. We train such two autoencoders for the clean and adversarial image domains (i.e., cross-domain) respectively to achieve image reconstructions and feature disentanglement, by using pairs of clean images and their Adversarial Examples (\textbf{AE}s) generated from available surrogate models via white-box attack methods. Eventually, in the black-box attack stage, DifAttack++ iteratively optimizes the AF according to the query feedback from the victim model until a successful AE is generated, while keeping the VF unaltered. Extensive experimental results demonstrate that our DifAttack++ leads to superior ASR and query efficiency than state-of-the-art methods, meanwhile exhibiting much better visual quality of AEs. The code is available at https://github.com/csjunjun/DifAttack.git.



## **12. Time-Frequency Jointed Imperceptible Adversarial Attack to Brainprint Recognition with Deep Learning Models**

cs.CR

This work is accepted by ICME 2024

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2403.10021v3) [paper-pdf](http://arxiv.org/pdf/2403.10021v3)

**Authors**: Hangjie Yi, Yuhang Ming, Dongjun Liu, Wanzeng Kong

**Abstract**: EEG-based brainprint recognition with deep learning models has garnered much attention in biometric identification. Yet, studies have indicated vulnerability to adversarial attacks in deep learning models with EEG inputs. In this paper, we introduce a novel adversarial attack method that jointly attacks time-domain and frequency-domain EEG signals by employing wavelet transform. Different from most existing methods which only target time-domain EEG signals, our method not only takes advantage of the time-domain attack's potent adversarial strength but also benefits from the imperceptibility inherent in frequency-domain attack, achieving a better balance between attack performance and imperceptibility. Extensive experiments are conducted in both white- and grey-box scenarios and the results demonstrate that our attack method achieves state-of-the-art attack performance on three datasets and three deep-learning models. In the meanwhile, the perturbations in the signals attacked by our method are barely perceptible to the human visual system.



## **13. Learning Robust 3D Representation from CLIP via Dual Denoising**

cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.00905v1) [paper-pdf](http://arxiv.org/pdf/2407.00905v1)

**Authors**: Shuqing Luo, Bowen Qu, Wei Gao

**Abstract**: In this paper, we explore a critical yet under-investigated issue: how to learn robust and well-generalized 3D representation from pre-trained vision language models such as CLIP. Previous works have demonstrated that cross-modal distillation can provide rich and useful knowledge for 3D data. However, like most deep learning models, the resultant 3D learning network is still vulnerable to adversarial attacks especially the iterative attack. In this work, we propose Dual Denoising, a novel framework for learning robust and well-generalized 3D representations from CLIP. It combines a denoising-based proxy task with a novel feature denoising network for 3D pre-training. Additionally, we propose utilizing parallel noise inference to enhance the generalization of point cloud features under cross domain settings. Experiments show that our model can effectively improve the representation learning performance and adversarial robustness of the 3D learning network under zero-shot settings without adversarial training. Our code is available at https://github.com/luoshuqing2001/Dual_Denoising.



## **14. GRACE: Graph-Regularized Attentive Convolutional Entanglement with Laplacian Smoothing for Robust DeepFake Video Detection**

cs.CV

Submitted to TPAMI 2024

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2406.19941v2) [paper-pdf](http://arxiv.org/pdf/2406.19941v2)

**Authors**: Chih-Chung Hsu, Shao-Ning Chen, Mei-Hsuan Wu, Yi-Fang Wang, Chia-Ming Lee, Yi-Shiuan Chou

**Abstract**: As DeepFake video manipulation techniques escalate, posing profound threats, the urgent need to develop efficient detection strategies is underscored. However, one particular issue lies with facial images being mis-detected, often originating from degraded videos or adversarial attacks, leading to unexpected temporal artifacts that can undermine the efficacy of DeepFake video detection techniques. This paper introduces a novel method for robust DeepFake video detection, harnessing the power of the proposed Graph-Regularized Attentive Convolutional Entanglement (GRACE) based on the graph convolutional network with graph Laplacian to address the aforementioned challenges. First, conventional Convolution Neural Networks are deployed to perform spatiotemporal features for the entire video. Then, the spatial and temporal features are mutually entangled by constructing a graph with sparse constraint, enforcing essential features of valid face images in the noisy face sequences remaining, thus augmenting stability and performance for DeepFake video detection. Furthermore, the Graph Laplacian prior is proposed in the graph convolutional network to remove the noise pattern in the feature space to further improve the performance. Comprehensive experiments are conducted to illustrate that our proposed method delivers state-of-the-art performance in DeepFake video detection under noisy face sequences. The source code is available at https://github.com/ming053l/GRACE.



## **15. A Two-Layer Blockchain Sharding Protocol Leveraging Safety and Liveness for Enhanced Performance**

cs.CR

The paper has been accepted to Network and Distributed System  Security (NDSS) Symposium 2024

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2310.11373v4) [paper-pdf](http://arxiv.org/pdf/2310.11373v4)

**Authors**: Yibin Xu, Jingyi Zheng, Boris Düdder, Tijs Slaats, Yongluan Zhou

**Abstract**: Sharding is essential for improving blockchain scalability. Existing protocols overlook diverse adversarial attacks, limiting transaction throughput. This paper presents Reticulum, a groundbreaking sharding protocol addressing this issue, boosting blockchain scalability.   Reticulum employs a two-phase approach, adapting transaction throughput based on runtime adversarial attacks. It comprises "control" and "process" shards in two layers. Process shards contain at least one trustworthy node, while control shards have a majority of trusted nodes. In the first phase, transactions are written to blocks and voted on by nodes in process shards. Unanimously accepted blocks are confirmed. In the second phase, blocks without unanimous acceptance are voted on by control shards. Blocks are accepted if the majority votes in favor, eliminating first-phase opponents and silent voters. Reticulum uses unanimous voting in the first phase, involving fewer nodes, enabling more parallel process shards. Control shards finalize decisions and resolve disputes.   Experiments confirm Reticulum's innovative design, providing high transaction throughput and robustness against various network attacks, outperforming existing sharding protocols for blockchain networks.



## **16. Fortify the Guardian, Not the Treasure: Resilient Adversarial Detectors**

cs.CV

**SubmitDate**: 2024-06-30    [abs](http://arxiv.org/abs/2404.12120v2) [paper-pdf](http://arxiv.org/pdf/2404.12120v2)

**Authors**: Raz Lapid, Almog Dubin, Moshe Sipper

**Abstract**: This paper presents RADAR-Robust Adversarial Detection via Adversarial Retraining-an approach designed to enhance the robustness of adversarial detectors against adaptive attacks, while maintaining classifier performance. An adaptive attack is one where the attacker is aware of the defenses and adapts their strategy accordingly. Our proposed method leverages adversarial training to reinforce the ability to detect attacks, without compromising clean accuracy. During the training phase, we integrate into the dataset adversarial examples, which were optimized to fool both the classifier and the adversarial detector, enabling the adversarial detector to learn and adapt to potential attack scenarios. Experimental evaluations on the CIFAR-10 and SVHN datasets demonstrate that our proposed algorithm significantly improves a detector's ability to accurately identify adaptive adversarial attacks -- without sacrificing clean accuracy.



## **17. Query-Efficient Hard-Label Black-Box Attack against Vision Transformers**

cs.CV

**SubmitDate**: 2024-06-29    [abs](http://arxiv.org/abs/2407.00389v1) [paper-pdf](http://arxiv.org/pdf/2407.00389v1)

**Authors**: Chao Zhou, Xiaowen Shi, Yuan-Gen Wang

**Abstract**: Recent studies have revealed that vision transformers (ViTs) face similar security risks from adversarial attacks as deep convolutional neural networks (CNNs). However, directly applying attack methodology on CNNs to ViTs has been demonstrated to be ineffective since the ViTs typically work on patch-wise encoding. This article explores the vulnerability of ViTs against adversarial attacks under a black-box scenario, and proposes a novel query-efficient hard-label adversarial attack method called AdvViT. Specifically, considering that ViTs are highly sensitive to patch modification, we propose to optimize the adversarial perturbation on the individual patches. To reduce the dimension of perturbation search space, we modify only a handful of low-frequency components of each patch. Moreover, we design a weight mask matrix for all patches to further optimize the perturbation on different regions of a whole image. We test six mainstream ViT backbones on the ImageNet-1k dataset. Experimental results show that compared with the state-of-the-art attacks on CNNs, our AdvViT achieves much lower $L_2$-norm distortion under the same query budget, sufficiently validating the vulnerability of ViTs against adversarial attacks.



## **18. DiffuseDef: Improved Robustness to Adversarial Attacks**

cs.CL

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2407.00248v1) [paper-pdf](http://arxiv.org/pdf/2407.00248v1)

**Authors**: Zhenhao Li, Marek Rei, Lucia Specia

**Abstract**: Pretrained language models have significantly advanced performance across various natural language processing tasks. However, adversarial attacks continue to pose a critical challenge to system built using these models, as they can be exploited with carefully crafted adversarial texts. Inspired by the ability of diffusion models to predict and reduce noise in computer vision, we propose a novel and flexible adversarial defense method for language classification tasks, DiffuseDef, which incorporates a diffusion layer as a denoiser between the encoder and the classifier. During inference, the adversarial hidden state is first combined with sampled noise, then denoised iteratively and finally ensembled to produce a robust text representation. By integrating adversarial training, denoising, and ensembling techniques, we show that DiffuseDef improves over different existing adversarial defense methods and achieves state-of-the-art performance against common adversarial attacks.



## **19. Deciphering the Definition of Adversarial Robustness for post-hoc OOD Detectors**

cs.CR

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.15104v3) [paper-pdf](http://arxiv.org/pdf/2406.15104v3)

**Authors**: Peter Lorenz, Mario Fernandez, Jens Müller, Ullrich Köthe

**Abstract**: Detecting out-of-distribution (OOD) inputs is critical for safely deploying deep learning models in real-world scenarios. In recent years, many OOD detectors have been developed, and even the benchmarking has been standardized, i.e. OpenOOD. The number of post-hoc detectors is growing fast and showing an option to protect a pre-trained classifier against natural distribution shifts, claiming to be ready for real-world scenarios. However, its efficacy in handling adversarial examples has been neglected in the majority of studies. This paper investigates the adversarial robustness of the 16 post-hoc detectors on several evasion attacks and discuss a roadmap towards adversarial defense in OOD detectors.



## **20. Stackelberg Games with $k$-Submodular Function under Distributional Risk-Receptiveness and Robustness**

math.OC

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.13023v3) [paper-pdf](http://arxiv.org/pdf/2406.13023v3)

**Authors**: Seonghun Park, Manish Bansal

**Abstract**: We study submodular optimization in adversarial context, applicable to machine learning problems such as feature selection using data susceptible to uncertainties and attacks. We focus on Stackelberg games between an attacker (or interdictor) and a defender where the attacker aims to minimize the defender's objective of maximizing a $k$-submodular function. We allow uncertainties arising from the success of attacks and inherent data noise, and address challenges due to incomplete knowledge of the probability distribution of random parameters. Specifically, we introduce Distributionally Risk-Averse $k$-Submodular Interdiction Problem (DRA $k$-SIP) and Distributionally Risk-Receptive $k$-Submodular Interdiction Problem (DRR $k$-SIP) along with finitely convergent exact algorithms for solving them. The DRA $k$-SIP solution allows risk-averse interdictor to develop robust strategies for real-world uncertainties. Conversely, DRR $k$-SIP solution suggests aggressive tactics for attackers, willing to embrace (distributional) risk to inflict maximum damage, identifying critical vulnerable components, which can be used for the defender's defensive strategies. The optimal values derived from both DRA $k$-SIP and DRR $k$-SIP offer a confidence interval-like range for the expected value of the defender's objective function, capturing distributional ambiguity. We conduct computational experiments using instances of feature selection and sensor placement problems, and Wisconsin breast cancer data and synthetic data, respectively.



## **21. Emotion Loss Attacking: Adversarial Attack Perception for Skeleton based on Multi-dimensional Features**

cs.CV

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19815v1) [paper-pdf](http://arxiv.org/pdf/2406.19815v1)

**Authors**: Feng Liu, Qing Xu, Qijian Zheng

**Abstract**: Adversarial attack on skeletal motion is a hot topic. However, existing researches only consider part of dynamic features when measuring distance between skeleton graph sequences, which results in poor imperceptibility. To this end, we propose a novel adversarial attack method to attack action recognizers for skeletal motions. Firstly, our method systematically proposes a dynamic distance function to measure the difference between skeletal motions. Meanwhile, we innovatively introduce emotional features for complementary information. In addition, we use Alternating Direction Method of Multipliers(ADMM) to solve the constrained optimization problem, which generates adversarial samples with better imperceptibility to deceive the classifiers. Experiments show that our method is effective on multiple action classifiers and datasets. When the perturbation magnitude measured by l norms is the same, the dynamic perturbations generated by our method are much lower than that of other methods. What's more, we are the first to prove the effectiveness of emotional features, and provide a new idea for measuring the distance between skeletal motions.



## **22. Deceptive Diffusion: Generating Synthetic Adversarial Examples**

cs.LG

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19807v1) [paper-pdf](http://arxiv.org/pdf/2406.19807v1)

**Authors**: Lucas Beerens, Catherine F. Higham, Desmond J. Higham

**Abstract**: We introduce the concept of deceptive diffusion -- training a generative AI model to produce adversarial images. Whereas a traditional adversarial attack algorithm aims to perturb an existing image to induce a misclassificaton, the deceptive diffusion model can create an arbitrary number of new, misclassified images that are not directly associated with training or test images. Deceptive diffusion offers the possibility of strengthening defence algorithms by providing adversarial training data at scale, including types of misclassification that are otherwise difficult to find. In our experiments, we also investigate the effect of training on a partially attacked data set. This highlights a new type of vulnerability for generative diffusion models: if an attacker is able to stealthily poison a portion of the training data, then the resulting diffusion model will generate a similar proportion of misleading outputs.



## **23. Backdoor Attack in Prompt-Based Continual Learning**

cs.LG

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19753v1) [paper-pdf](http://arxiv.org/pdf/2406.19753v1)

**Authors**: Trang Nguyen, Anh Tran, Nhat Ho

**Abstract**: Prompt-based approaches offer a cutting-edge solution to data privacy issues in continual learning, particularly in scenarios involving multiple data suppliers where long-term storage of private user data is prohibited. Despite delivering state-of-the-art performance, its impressive remembering capability can become a double-edged sword, raising security concerns as it might inadvertently retain poisoned knowledge injected during learning from private user data. Following this insight, in this paper, we expose continual learning to a potential threat: backdoor attack, which drives the model to follow a desired adversarial target whenever a specific trigger is present while still performing normally on clean samples. We highlight three critical challenges in executing backdoor attacks on incremental learners and propose corresponding solutions: (1) \emph{Transferability}: We employ a surrogate dataset and manipulate prompt selection to transfer backdoor knowledge to data from other suppliers; (2) \emph{Resiliency}: We simulate static and dynamic states of the victim to ensure the backdoor trigger remains robust during intense incremental learning processes; and (3) \emph{Authenticity}: We apply binary cross-entropy loss as an anti-cheating factor to prevent the backdoor trigger from devolving into adversarial noise. Extensive experiments across various benchmark datasets and continual learners validate our continual backdoor framework, achieving up to $100\%$ attack success rate, with further ablation studies confirming our contributions' effectiveness.



## **24. Steering cooperation: Adversarial attacks on prisoner's dilemma in complex networks**

physics.soc-ph

14 pages, 4 figures

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19692v1) [paper-pdf](http://arxiv.org/pdf/2406.19692v1)

**Authors**: Kazuhiro Takemoto

**Abstract**: This study examines the application of adversarial attack concepts to control the evolution of cooperation in the prisoner's dilemma game in complex networks. Specifically, it proposes a simple adversarial attack method that drives players' strategies towards a target state by adding small perturbations to social networks. The proposed method is evaluated on both model and real-world networks. Numerical simulations demonstrate that the proposed method can effectively promote cooperation with significantly smaller perturbations compared to other techniques. Additionally, this study shows that adversarial attacks can also be useful in inhibiting cooperation (promoting defection). The findings reveal that adversarial attacks on social networks can be potent tools for both promoting and inhibiting cooperation, opening new possibilities for controlling cooperative behavior in social systems while also highlighting potential risks.



## **25. IDT: Dual-Task Adversarial Attacks for Privacy Protection**

cs.CL

28 pages, 1 figure

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19642v1) [paper-pdf](http://arxiv.org/pdf/2406.19642v1)

**Authors**: Pedro Faustini, Shakila Mahjabin Tonni, Annabelle McIver, Qiongkai Xu, Mark Dras

**Abstract**: Natural language processing (NLP) models may leak private information in different ways, including membership inference, reconstruction or attribute inference attacks. Sensitive information may not be explicit in the text, but hidden in underlying writing characteristics. Methods to protect privacy can involve using representations inside models that are demonstrated not to detect sensitive attributes or -- for instance, in cases where users might not trust a model, the sort of scenario of interest here -- changing the raw text before models can have access to it. The goal is to rewrite text to prevent someone from inferring a sensitive attribute (e.g. the gender of the author, or their location by the writing style) whilst keeping the text useful for its original intention (e.g. the sentiment of a product review). The few works tackling this have focused on generative techniques. However, these often create extensively different texts from the original ones or face problems such as mode collapse. This paper explores a novel adaptation of adversarial attack techniques to manipulate a text to deceive a classifier w.r.t one task (privacy) whilst keeping the predictions of another classifier trained for another task (utility) unchanged. We propose IDT, a method that analyses predictions made by auxiliary and interpretable models to identify which tokens are important to change for the privacy task, and which ones should be kept for the utility task. We evaluate different datasets for NLP suitable for different tasks. Automatic and human evaluations show that IDT retains the utility of text, while also outperforming existing methods when deceiving a classifier w.r.t privacy task.



## **26. Data-Driven Lipschitz Continuity: A Cost-Effective Approach to Improve Adversarial Robustness**

cs.LG

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19622v1) [paper-pdf](http://arxiv.org/pdf/2406.19622v1)

**Authors**: Erh-Chung Chen, Pin-Yu Chen, I-Hsin Chung, Che-Rung Lee

**Abstract**: The security and robustness of deep neural networks (DNNs) have become increasingly concerning. This paper aims to provide both a theoretical foundation and a practical solution to ensure the reliability of DNNs. We explore the concept of Lipschitz continuity to certify the robustness of DNNs against adversarial attacks, which aim to mislead the network with adding imperceptible perturbations into inputs. We propose a novel algorithm that remaps the input domain into a constrained range, reducing the Lipschitz constant and potentially enhancing robustness. Unlike existing adversarially trained models, where robustness is enhanced by introducing additional examples from other datasets or generative models, our method is almost cost-free as it can be integrated with existing models without requiring re-training. Experimental results demonstrate the generalizability of our method, as it can be combined with various models and achieve enhancements in robustness. Furthermore, our method achieves the best robust accuracy for CIFAR10, CIFAR100, and ImageNet datasets on the RobustBench leaderboard.



## **27. Zero-Query Adversarial Attack on Black-box Automatic Speech Recognition Systems**

cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.19311v1) [paper-pdf](http://arxiv.org/pdf/2406.19311v1)

**Authors**: Zheng Fang, Tao Wang, Lingchen Zhao, Shenyi Zhang, Bowen Li, Yunjie Ge, Qi Li, Chao Shen, Qian Wang

**Abstract**: In recent years, extensive research has been conducted on the vulnerability of ASR systems, revealing that black-box adversarial example attacks pose significant threats to real-world ASR systems. However, most existing black-box attacks rely on queries to the target ASRs, which is impractical when queries are not permitted. In this paper, we propose ZQ-Attack, a transfer-based adversarial attack on ASR systems in the zero-query black-box setting. Through a comprehensive review and categorization of modern ASR technologies, we first meticulously select surrogate ASRs of diverse types to generate adversarial examples. Following this, ZQ-Attack initializes the adversarial perturbation with a scaled target command audio, rendering it relatively imperceptible while maintaining effectiveness. Subsequently, to achieve high transferability of adversarial perturbations, we propose a sequential ensemble optimization algorithm, which iteratively optimizes the adversarial perturbation on each surrogate model, leveraging collaborative information from other models. We conduct extensive experiments to evaluate ZQ-Attack. In the over-the-line setting, ZQ-Attack achieves a 100% success rate of attack (SRoA) with an average signal-to-noise ratio (SNR) of 21.91dB on 4 online speech recognition services, and attains an average SRoA of 100% and SNR of 19.67dB on 16 open-source ASRs. For commercial intelligent voice control devices, ZQ-Attack also achieves a 100% SRoA with an average SNR of 15.77dB in the over-the-air setting.



## **28. GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts**

cs.AI

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2309.10253v4) [paper-pdf](http://arxiv.org/pdf/2309.10253v4)

**Authors**: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

**Abstract**: Large language models (LLMs) have recently experienced tremendous popularity and are widely used from casual conversations to AI-driven programming. However, despite their considerable success, LLMs are not entirely reliable and can give detailed guidance on how to conduct harmful or illegal activities. While safety measures can reduce the risk of such outputs, adversarial jailbreak attacks can still exploit LLMs to produce harmful content. These jailbreak templates are typically manually crafted, making large-scale testing challenging.   In this paper, we introduce GPTFuzz, a novel black-box jailbreak fuzzing framework inspired by the AFL fuzzing framework. Instead of manual engineering, GPTFuzz automates the generation of jailbreak templates for red-teaming LLMs. At its core, GPTFuzz starts with human-written templates as initial seeds, then mutates them to produce new templates. We detail three key components of GPTFuzz: a seed selection strategy for balancing efficiency and variability, mutate operators for creating semantically equivalent or similar sentences, and a judgment model to assess the success of a jailbreak attack.   We evaluate GPTFuzz against various commercial and open-source LLMs, including ChatGPT, LLaMa-2, and Vicuna, under diverse attack scenarios. Our results indicate that GPTFuzz consistently produces jailbreak templates with a high success rate, surpassing human-crafted templates. Remarkably, GPTFuzz achieves over 90% attack success rates against ChatGPT and Llama-2 models, even with suboptimal initial seed templates. We anticipate that GPTFuzz will be instrumental for researchers and practitioners in examining LLM robustness and will encourage further exploration into enhancing LLM safety.



## **29. Spiking Convolutional Neural Networks for Text Classification**

cs.NE

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.19230v1) [paper-pdf](http://arxiv.org/pdf/2406.19230v1)

**Authors**: Changze Lv, Jianhan Xu, Xiaoqing Zheng

**Abstract**: Spiking neural networks (SNNs) offer a promising pathway to implement deep neural networks (DNNs) in a more energy-efficient manner since their neurons are sparsely activated and inferences are event-driven. However, there have been very few works that have demonstrated the efficacy of SNNs in language tasks partially because it is non-trivial to represent words in the forms of spikes and to deal with variable-length texts by SNNs. This work presents a "conversion + fine-tuning" two-step method for training SNNs for text classification and proposes a simple but effective way to encode pre-trained word embeddings as spike trains. We show empirically that after fine-tuning with surrogate gradients, the converted SNNs achieve comparable results to their DNN counterparts with much less energy consumption across multiple datasets for both English and Chinese. We also show that such SNNs are more robust to adversarial attacks than DNNs.



## **30. Understanding the Security Benefits and Overheads of Emerging Industry Solutions to DRAM Read Disturbance**

cs.CR

To appear in DRAMSec 2024

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.19094v1) [paper-pdf](http://arxiv.org/pdf/2406.19094v1)

**Authors**: Oğuzhan Canpolat, A. Giray Yağlıkçı, Geraldo F. Oliveira, Ataberk Olgun, Oğuz Ergin, Onur Mutlu

**Abstract**: We present the first rigorous security, performance, energy, and cost analyses of the state-of-the-art on-DRAM-die read disturbance mitigation method, Per Row Activation Counting (PRAC), described in JEDEC DDR5 specification's April 2024 update. Unlike prior state-of-the-art that advises the memory controller to periodically issue refresh management (RFM) commands, which provides the DRAM chip with time to perform refreshes, PRAC introduces a new back-off signal. PRAC's back-off signal propagates from the DRAM chip to the memory controller and forces the memory controller to 1) stop serving requests and 2) issue RFM commands. As a result, RFM commands are issued when needed as opposed to periodically, reducing RFM's overheads. We analyze PRAC in four steps. First, we define an adversarial access pattern that represents the worst-case for PRAC's security. Second, we investigate PRAC's configurations and security implications. Our analyses show that PRAC can be configured for secure operation as long as no bitflip occurs before accessing a memory location 10 times. Third, we evaluate the performance impact of PRAC and compare it against prior works using Ramulator 2.0. Our analysis shows that while PRAC incurs less than 13.4% performance overhead for today's DRAM chips, its performance overheads can reach up to 63.2% for future DRAM chips that are more vulnerable to read disturbance bitflips. Fourth, we define an availability adversarial access pattern that exacerbates PRAC's performance overhead to perform a memory performance attack, demonstrating that such an adversarial pattern can hog up to 79% of DRAM throughput and degrade system throughput by up to 65%. We discuss PRAC's implications on future systems and foreshadow future research directions. To aid future research, we open-source our implementations and scripts at https://github.com/CMU-SAFARI/ramulator2.



## **31. Intriguing Properties of Adversarial ML Attacks in the Problem Space [Extended Version]**

cs.CR

This arXiv version (v3) corresponds to an extended version

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/1911.02142v3) [paper-pdf](http://arxiv.org/pdf/1911.02142v3)

**Authors**: Jacopo Cortellazzi, Feargus Pendlebury, Daniel Arp, Erwin Quiring, Fabio Pierazzi, Lorenzo Cavallaro

**Abstract**: Recent research efforts on adversarial machine learning (ML) have investigated problem-space attacks, focusing on the generation of real evasive objects in domains where, unlike images, there is no clear inverse mapping to the feature space (e.g., software). However, the design, comparison, and real-world implications of problem-space attacks remain underexplored. This article makes three major contributions. Firstly, we propose a general formalization for adversarial ML evasion attacks in the problem-space, which includes the definition of a comprehensive set of constraints on available transformations, preserved semantics, absent artifacts, and plausibility. We shed light on the relationship between feature space and problem space, and we introduce the concept of side-effect features as the by-product of the inverse feature-mapping problem. This enables us to define and prove necessary and sufficient conditions for the existence of problem-space attacks. Secondly, building on our general formalization, we propose a novel problem-space attack on Android malware that overcomes past limitations in terms of semantics and artifacts. We have tested our approach on a dataset with 150K Android apps from 2016 and 2018 which show the practical feasibility of evading a state-of-the-art malware classifier along with its hardened version. Thirdly, we explore the effectiveness of adversarial training as a possible approach to enforce robustness against adversarial samples, evaluating its effectiveness on the considered machine learning models under different scenarios. Our results demonstrate that "adversarial-malware as a service" is a realistic threat, as we automatically generate thousands of realistic and inconspicuous adversarial applications at scale, where on average it takes only a few minutes to generate an adversarial instance.



## **32. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

cs.CV

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.18849v1) [paper-pdf](http://arxiv.org/pdf/2406.18849v1)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 8 advanced open-source LVLMs with 10 checkpoints are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released in \url{https://github.com/Benchmark-Dysca/Dysca}.



## **33. A Zero Auxiliary Knowledge Membership Inference Attack on Aggregate Location Data**

cs.CR

To be published in PETS 2024

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18671v1) [paper-pdf](http://arxiv.org/pdf/2406.18671v1)

**Authors**: Vincent Guan, Florent Guépin, Ana-Maria Cretu, Yves-Alexandre de Montjoye

**Abstract**: Location data is frequently collected from populations and shared in aggregate form to guide policy and decision making. However, the prevalence of aggregated data also raises the privacy concern of membership inference attacks (MIAs). MIAs infer whether an individual's data contributed to the aggregate release. Although effective MIAs have been developed for aggregate location data, these require access to an extensive auxiliary dataset of individual traces over the same locations, which are collected from a similar population. This assumption is often impractical given common privacy practices surrounding location data. To measure the risk of an MIA performed by a realistic adversary, we develop the first Zero Auxiliary Knowledge (ZK) MIA on aggregate location data, which eliminates the need for an auxiliary dataset of real individual traces. Instead, we develop a novel synthetic approach, such that suitable synthetic traces are generated from the released aggregate. We also develop methods to correct for bias and noise, to show that our synthetic-based attack is still applicable when privacy mechanisms are applied prior to release. Using two large-scale location datasets, we demonstrate that our ZK MIA matches the state-of-the-art Knock-Knock (KK) MIA across a wide range of settings, including popular implementations of differential privacy (DP) and suppression of small counts. Furthermore, we show that ZK MIA remains highly effective even when the adversary only knows a small fraction (10%) of their target's location history. This demonstrates that effective MIAs can be performed by realistic adversaries, highlighting the need for strong DP protection.



## **34. WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models**

cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18510v1) [paper-pdf](http://arxiv.org/pdf/2406.18510v1)

**Authors**: Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, Nouha Dziri

**Abstract**: We introduce WildTeaming, an automatic LLM safety red-teaming framework that mines in-the-wild user-chatbot interactions to discover 5.7K unique clusters of novel jailbreak tactics, and then composes multiple tactics for systematic exploration of novel jailbreaks. Compared to prior work that performed red-teaming via recruited human workers, gradient-based optimization, or iterative revision with LLMs, our work investigates jailbreaks from chatbot users who were not specifically instructed to break the system. WildTeaming reveals previously unidentified vulnerabilities of frontier LLMs, resulting in up to 4.6x more diverse and successful adversarial attacks compared to state-of-the-art jailbreak methods.   While many datasets exist for jailbreak evaluation, very few open-source datasets exist for jailbreak training, as safety training data has been closed even when model weights are open. With WildTeaming we create WildJailbreak, a large-scale open-source synthetic safety dataset with 262K vanilla (direct request) and adversarial (complex jailbreak) prompt-response pairs. To mitigate exaggerated safety behaviors, WildJailbreak provides two contrastive types of queries: 1) harmful queries (vanilla & adversarial) and 2) benign queries that resemble harmful queries in form but contain no harm. As WildJailbreak considerably upgrades the quality and scale of existing safety resources, it uniquely enables us to examine the scaling effects of data and the interplay of data properties and model capabilities during safety training. Through extensive experiments, we identify the training properties that enable an ideal balance of safety behaviors: appropriate safeguarding without over-refusal, effective handling of vanilla and adversarial queries, and minimal, if any, decrease in general capabilities. All components of WildJailbeak contribute to achieving balanced safety behaviors of models.



## **35. WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs**

cs.CL

First two authors contributed equally. Third and fourth authors  contributed equally

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18495v1) [paper-pdf](http://arxiv.org/pdf/2406.18495v1)

**Authors**: Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, Nouha Dziri

**Abstract**: We introduce WildGuard -- an open, light-weight moderation tool for LLM safety that achieves three goals: (1) identifying malicious intent in user prompts, (2) detecting safety risks of model responses, and (3) determining model refusal rate. Together, WildGuard serves the increasing needs for automatic safety moderation and evaluation of LLM interactions, providing a one-stop tool with enhanced accuracy and broad coverage across 13 risk categories. While existing open moderation tools such as Llama-Guard2 score reasonably well in classifying straightforward model interactions, they lag far behind a prompted GPT-4, especially in identifying adversarial jailbreaks and in evaluating models' refusals, a key measure for evaluating safety behaviors in model responses.   To address these challenges, we construct WildGuardMix, a large-scale and carefully balanced multi-task safety moderation dataset with 92K labeled examples that cover vanilla (direct) prompts and adversarial jailbreaks, paired with various refusal and compliance responses. WildGuardMix is a combination of WildGuardTrain, the training data of WildGuard, and WildGuardTest, a high-quality human-annotated moderation test set with 5K labeled items covering broad risk scenarios. Through extensive evaluations on WildGuardTest and ten existing public benchmarks, we show that WildGuard establishes state-of-the-art performance in open-source safety moderation across all the three tasks compared to ten strong existing open-source moderation models (e.g., up to 26.4% improvement on refusal detection). Importantly, WildGuard matches and sometimes exceeds GPT-4 performance (e.g., up to 3.9% improvement on prompt harmfulness identification). WildGuard serves as a highly effective safety moderator in an LLM interface, reducing the success rate of jailbreak attacks from 79.8% to 2.4%.



## **36. Enhancing Federated Learning with Adaptive Differential Privacy and Priority-Based Aggregation**

cs.LG

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18491v1) [paper-pdf](http://arxiv.org/pdf/2406.18491v1)

**Authors**: Mahtab Talaei, Iman Izadi

**Abstract**: Federated learning (FL), a novel branch of distributed machine learning (ML), develops global models through a private procedure without direct access to local datasets. However, it is still possible to access the model updates (gradient updates of deep neural networks) transferred between clients and servers, potentially revealing sensitive local information to adversaries using model inversion attacks. Differential privacy (DP) offers a promising approach to addressing this issue by adding noise to the parameters. On the other hand, heterogeneities in data structure, storage, communication, and computational capabilities of devices can cause convergence problems and delays in developing the global model. A personalized weighted averaging of local parameters based on the resources of each device can yield a better aggregated model in each round. In this paper, to efficiently preserve privacy, we propose a personalized DP framework that injects noise based on clients' relative impact factors and aggregates parameters while considering heterogeneities and adjusting properties. To fulfill the DP requirements, we first analyze the convergence boundary of the FL algorithm when impact factors are personalized and fixed throughout the learning process. We then further study the convergence property considering time-varying (adaptive) impact factors.



## **37. MultiAgent Collaboration Attack: Investigating Adversarial Attacks in Large Language Model Collaborations via Debate**

cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.14711v2) [paper-pdf](http://arxiv.org/pdf/2406.14711v2)

**Authors**: Alfonso Amayuelas, Xianjun Yang, Antonis Antoniades, Wenyue Hua, Liangming Pan, William Wang

**Abstract**: Large Language Models (LLMs) have shown exceptional results on current benchmarks when working individually. The advancement in their capabilities, along with a reduction in parameter size and inference times, has facilitated the use of these models as agents, enabling interactions among multiple models to execute complex tasks. Such collaborations offer several advantages, including the use of specialized models (e.g. coding), improved confidence through multiple computations, and enhanced divergent thinking, leading to more diverse outputs. Thus, the collaborative use of language models is expected to grow significantly in the coming years. In this work, we evaluate the behavior of a network of models collaborating through debate under the influence of an adversary. We introduce pertinent metrics to assess the adversary's effectiveness, focusing on system accuracy and model agreement. Our findings highlight the importance of a model's persuasive ability in influencing others. Additionally, we explore inference-time methods to generate more compelling arguments and evaluate the potential of prompt-based mitigation as a defensive strategy.



## **38. Detecting Brittle Decisions for Free: Leveraging Margin Consistency in Deep Robust Classifiers**

cs.LG

11 pages, 7 figures, 2 tables, 1 algorithm

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18451v1) [paper-pdf](http://arxiv.org/pdf/2406.18451v1)

**Authors**: Jonas Ngnawé, Sabyasachi Sahoo, Yann Pequignot, Frédéric Precioso, Christian Gagné

**Abstract**: Despite extensive research on adversarial training strategies to improve robustness, the decisions of even the most robust deep learning models can still be quite sensitive to imperceptible perturbations, creating serious risks when deploying them for high-stakes real-world applications. While detecting such cases may be critical, evaluating a model's vulnerability at a per-instance level using adversarial attacks is computationally too intensive and unsuitable for real-time deployment scenarios. The input space margin is the exact score to detect non-robust samples and is intractable for deep neural networks. This paper introduces the concept of margin consistency -- a property that links the input space margins and the logit margins in robust models -- for efficient detection of vulnerable samples. First, we establish that margin consistency is a necessary and sufficient condition to use a model's logit margin as a score for identifying non-robust samples. Next, through comprehensive empirical analysis of various robustly trained models on CIFAR10 and CIFAR100 datasets, we show that they indicate strong margin consistency with a strong correlation between their input space margins and the logit margins. Then, we show that we can effectively use the logit margin to confidently detect brittle decisions with such models and accurately estimate robust accuracy on an arbitrarily large test set by estimating the input margins only on a small subset. Finally, we address cases where the model is not sufficiently margin-consistent by learning a pseudo-margin from the feature representation. Our findings highlight the potential of leveraging deep representations to efficiently assess adversarial vulnerability in deployment scenarios.



## **39. Are AI-Generated Text Detectors Robust to Adversarial Perturbations?**

cs.CL

Accepted to ACL 2024 main conference

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.01179v2) [paper-pdf](http://arxiv.org/pdf/2406.01179v2)

**Authors**: Guanhua Huang, Yuchen Zhang, Zhe Li, Yongjian You, Mingze Wang, Zhouwang Yang

**Abstract**: The widespread use of large language models (LLMs) has sparked concerns about the potential misuse of AI-generated text, as these models can produce content that closely resembles human-generated text. Current detectors for AI-generated text (AIGT) lack robustness against adversarial perturbations, with even minor changes in characters or words causing a reversal in distinguishing between human-created and AI-generated text. This paper investigates the robustness of existing AIGT detection methods and introduces a novel detector, the Siamese Calibrated Reconstruction Network (SCRN). The SCRN employs a reconstruction network to add and remove noise from text, extracting a semantic representation that is robust to local perturbations. We also propose a siamese calibration technique to train the model to make equally confidence predictions under different noise, which improves the model's robustness against adversarial perturbations. Experiments on four publicly available datasets show that the SCRN outperforms all baseline methods, achieving 6.5\%-18.25\% absolute accuracy improvement over the best baseline method under adversarial attacks. Moreover, it exhibits superior generalizability in cross-domain, cross-genre, and mixed-source scenarios. The code is available at \url{https://github.com/CarlanLark/Robust-AIGC-Detector}.



## **40. SUB-PLAY: Adversarial Policies against Partially Observed Multi-Agent Reinforcement Learning Systems**

cs.LG

To appear in the ACM Conference on Computer and Communications  Security (CCS'24), October 14-18, 2024, Salt Lake City, UT, USA

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2402.03741v3) [paper-pdf](http://arxiv.org/pdf/2402.03741v3)

**Authors**: Oubo Ma, Yuwen Pu, Linkang Du, Yang Dai, Ruo Wang, Xiaolei Liu, Yingcai Wu, Shouling Ji

**Abstract**: Recent advancements in multi-agent reinforcement learning (MARL) have opened up vast application prospects, such as swarm control of drones, collaborative manipulation by robotic arms, and multi-target encirclement. However, potential security threats during the MARL deployment need more attention and thorough investigation. Recent research reveals that attackers can rapidly exploit the victim's vulnerabilities, generating adversarial policies that result in the failure of specific tasks. For instance, reducing the winning rate of a superhuman-level Go AI to around 20%. Existing studies predominantly focus on two-player competitive environments, assuming attackers possess complete global state observation.   In this study, we unveil, for the first time, the capability of attackers to generate adversarial policies even when restricted to partial observations of the victims in multi-agent competitive environments. Specifically, we propose a novel black-box attack (SUB-PLAY) that incorporates the concept of constructing multiple subgames to mitigate the impact of partial observability and suggests sharing transitions among subpolicies to improve attackers' exploitative ability. Extensive evaluations demonstrate the effectiveness of SUB-PLAY under three typical partial observability limitations. Visualization results indicate that adversarial policies induce significantly different activations of the victims' policy networks. Furthermore, we evaluate three potential defenses aimed at exploring ways to mitigate security threats posed by adversarial policies, providing constructive recommendations for deploying MARL in competitive environments.



## **41. Artificial Immune System of Secure Face Recognition Against Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18144v1) [paper-pdf](http://arxiv.org/pdf/2406.18144v1)

**Authors**: Min Ren, Yunlong Wang, Yuhao Zhu, Yongzhen Huang, Zhenan Sun, Qi Li, Tieniu Tan

**Abstract**: Insect production for food and feed presents a promising supplement to ensure food safety and address the adverse impacts of agriculture on climate and environment in the future. However, optimisation is required for insect production to realise its full potential. This can be by targeted improvement of traits of interest through selective breeding, an approach which has so far been underexplored and underutilised in insect farming. Here we present a comprehensive review of the selective breeding framework in the context of insect production. We systematically evaluate adjustments of selective breeding techniques to the realm of insects and highlight the essential components integral to the breeding process. The discussion covers every step of a conventional breeding scheme, such as formulation of breeding objectives, phenotyping, estimation of genetic parameters and breeding values, selection of appropriate breeding strategies, and mitigation of issues associated with genetic diversity depletion and inbreeding. This review combines knowledge from diverse disciplines, bridging the gap between animal breeding, quantitative genetics, evolutionary biology, and entomology, offering an integrated view of the insect breeding research area and uniting knowledge which has previously remained scattered across diverse fields of expertise.



## **42. Breaking the Barrier: Enhanced Utility and Robustness in Smoothed DRL Agents**

cs.LG

Published in ICML 2024

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18062v1) [paper-pdf](http://arxiv.org/pdf/2406.18062v1)

**Authors**: Chung-En Sun, Sicun Gao, Tsui-Wei Weng

**Abstract**: Robustness remains a paramount concern in deep reinforcement learning (DRL), with randomized smoothing emerging as a key technique for enhancing this attribute. However, a notable gap exists in the performance of current smoothed DRL agents, often characterized by significantly low clean rewards and weak robustness. In response to this challenge, our study introduces innovative algorithms aimed at training effective smoothed robust DRL agents. We propose S-DQN and S-PPO, novel approaches that demonstrate remarkable improvements in clean rewards, empirical robustness, and robustness guarantee across standard RL benchmarks. Notably, our S-DQN and S-PPO agents not only significantly outperform existing smoothed agents by an average factor of $2.16\times$ under the strongest attack, but also surpass previous robustly-trained agents by an average factor of $2.13\times$. This represents a significant leap forward in the field. Furthermore, we introduce Smoothed Attack, which is $1.89\times$ more effective in decreasing the rewards of smoothed agents than existing adversarial attacks.



## **43. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

cs.CV

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2312.01886v3) [paper-pdf](http://arxiv.org/pdf/2312.01886v3)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical targeted attack scenario that the adversary can only know the vision encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed \textsc{InstructTA}) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same vision encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability with instruction tuning, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from GPT-4. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability. The code is available at https://github.com/xunguangwang/InstructTA.



## **44. Diffusion-based Adversarial Purification for Intrusion Detection**

cs.CR

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17606v1) [paper-pdf](http://arxiv.org/pdf/2406.17606v1)

**Authors**: Mohamed Amine Merzouk, Erwan Beurier, Reda Yaich, Nora Boulahia-Cuppens, Frédéric Cuppens

**Abstract**: The escalating sophistication of cyberattacks has encouraged the integration of machine learning techniques in intrusion detection systems, but the rise of adversarial examples presents a significant challenge. These crafted perturbations mislead ML models, enabling attackers to evade detection or trigger false alerts. As a reaction, adversarial purification has emerged as a compelling solution, particularly with diffusion models showing promising results. However, their purification potential remains unexplored in the context of intrusion detection. This paper demonstrates the effectiveness of diffusion models in purifying adversarial examples in network intrusion detection. Through a comprehensive analysis of the diffusion parameters, we identify optimal configurations maximizing adversarial robustness with minimal impact on normal performance. Importantly, this study reveals insights into the relationship between diffusion noise and diffusion steps, representing a novel contribution to the field. Our experiments are carried out on two datasets and against 5 adversarial attacks. The implementation code is publicly available.



## **45. Treatment of Statistical Estimation Problems in Randomized Smoothing for Adversarial Robustness**

stat.ML

comments are welcome

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17830v1) [paper-pdf](http://arxiv.org/pdf/2406.17830v1)

**Authors**: Vaclav Voracek

**Abstract**: Randomized smoothing is a popular certified defense against adversarial attacks. In its essence, we need to solve a problem of statistical estimation which is usually very time-consuming since we need to perform numerous (usually $10^5$) forward passes of the classifier for every point to be certified. In this paper, we review the statistical estimation problems for randomized smoothing to find out if the computational burden is necessary. In particular, we consider the (standard) task of adversarial robustness where we need to decide if a point is robust at a certain radius or not using as few samples as possible while maintaining statistical guarantees. We present estimation procedures employing confidence sequences enjoying the same statistical guarantees as the standard methods, with the optimal sample complexities for the estimation task and empirically demonstrate their good performance. Additionally, we provide a randomized version of Clopper-Pearson confidence intervals resulting in strictly stronger certificates.



## **46. Detection of Synthetic Face Images: Accuracy, Robustness, Generalization**

cs.CV

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17547v1) [paper-pdf](http://arxiv.org/pdf/2406.17547v1)

**Authors**: Nela Petrzelkova, Jan Cech

**Abstract**: An experimental study on detecting synthetic face images is presented. We collected a dataset, called FF5, of five fake face image generators, including recent diffusion models. We find that a simple model trained on a specific image generator can achieve near-perfect accuracy in separating synthetic and real images. The model handles common image distortions (reduced resolution, compression) by using data augmentation. Moreover, partial manipulations, where synthetic images are blended into real ones by inpainting, are identified and the area of the manipulation is localized by a simple model of YOLO architecture. However, the model turned out to be vulnerable to adversarial attacks and does not generalize to unseen generators. Failure to generalize to detect images produced by a newer generator also occurs for recent state-of-the-art methods, which we tested on Realistic Vision, a fine-tuned version of StabilityAI's Stable Diffusion image generator.



## **47. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

cs.CL

Repo: https://github.com/wjfu99/MIA-LLMs

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2311.06062v3) [paper-pdf](http://arxiv.org/pdf/2311.06062v3)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.



## **48. TSynD: Targeted Synthetic Data Generation for Enhanced Medical Image Classification**

cs.CV

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17473v1) [paper-pdf](http://arxiv.org/pdf/2406.17473v1)

**Authors**: Joshua Niemeijer, Jan Ehrhardt, Hristina Uzunova, Heinz Handels

**Abstract**: The usage of medical image data for the training of large-scale machine learning approaches is particularly challenging due to its scarce availability and the costly generation of data annotations, typically requiring the engagement of medical professionals. The rapid development of generative models allows towards tackling this problem by leveraging large amounts of realistic synthetically generated data for the training process. However, randomly choosing synthetic samples, might not be an optimal strategy.   In this work, we investigate the targeted generation of synthetic training data, in order to improve the accuracy and robustness of image classification. Therefore, our approach aims to guide the generative model to synthesize data with high epistemic uncertainty, since large measures of epistemic uncertainty indicate underrepresented data points in the training set. During the image generation we feed images reconstructed by an auto encoder into the classifier and compute the mutual information over the class-probability distribution as a measure for uncertainty.We alter the feature space of the autoencoder through an optimization process with the objective of maximizing the classifier uncertainty on the decoded image. By training on such data we improve the performance and robustness against test time data augmentations and adversarial attacks on several classifications tasks.



## **49. Low-Cost Privacy-Aware Decentralized Learning**

cs.LG

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2403.11795v2) [paper-pdf](http://arxiv.org/pdf/2403.11795v2)

**Authors**: Sayan Biswas, Davide Frey, Romaric Gaudel, Anne-Marie Kermarrec, Dimitri Lerévérend, Rafael Pires, Rishi Sharma, François Taïani

**Abstract**: This paper introduces ZIP-DL, a novel privacy-aware decentralized learning (DL) algorithm that exploits correlated noise to provide strong privacy protection against a local adversary while yielding efficient convergence guarantees for a low communication cost. The progressive neutralization of the added noise during the distributed aggregation process results in ZIP-DL fostering a high model accuracy under privacy guarantees. ZIP-DL further uses a single communication round between each gradient descent, thus minimizing communication overhead. We provide theoretical guarantees for both convergence speed and privacy guarantees, thereby making ZIP-DL applicable to practical scenarios. Our extensive experimental study shows that ZIP-DL significantly outperforms the state-of-the-art in terms of vulnerability/accuracy trade-off. In particular, ZIP-DL (i) reduces the efficacy of linkability attacks by up to 52 percentage points compared to baseline DL, (ii) improves accuracy by up to 37 percent w.r.t. the state-of-the-art privacy-preserving mechanism operating under the same threat model as ours, when configured to provide the same protection against membership inference attacks, and (iii) reduces communication by up to 10.5x against the same competitor for the same level of protection.



## **50. CuDA2: An approach for Incorporating Traitor Agents into Cooperative Multi-Agent Systems**

cs.LG

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17425v1) [paper-pdf](http://arxiv.org/pdf/2406.17425v1)

**Authors**: Zhen Chen, Yong Liao, Youpeng Zhao, Zipeng Dai, Jian Zhao

**Abstract**: Cooperative Multi-Agent Reinforcement Learning (CMARL) strategies are well known to be vulnerable to adversarial perturbations. Previous works on adversarial attacks have primarily focused on white-box attacks that directly perturb the states or actions of victim agents, often in scenarios with a limited number of attacks. However, gaining complete access to victim agents in real-world environments is exceedingly difficult. To create more realistic adversarial attacks, we introduce a novel method that involves injecting traitor agents into the CMARL system. We model this problem as a Traitor Markov Decision Process (TMDP), where traitors cannot directly attack the victim agents but can influence their formation or positioning through collisions. In TMDP, traitors are trained using the same MARL algorithm as the victim agents, with their reward function set as the negative of the victim agents' reward. Despite this, the training efficiency for traitors remains low because it is challenging for them to directly associate their actions with the victim agents' rewards. To address this issue, we propose the Curiosity-Driven Adversarial Attack (CuDA2) framework. CuDA2 enhances the efficiency and aggressiveness of attacks on the specified victim agents' policies while maintaining the optimal policy invariance of the traitors. Specifically, we employ a pre-trained Random Network Distillation (RND) module, where the extra reward generated by the RND module encourages traitors to explore states unencountered by the victim agents. Extensive experiments on various scenarios from SMAC demonstrate that our CuDA2 framework offers comparable or superior adversarial attack capabilities compared to other baselines.



