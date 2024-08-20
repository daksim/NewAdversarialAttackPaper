# Latest Adversarial Attack Papers
**update at 2024-08-20 11:03:06**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Fight Perturbations with Perturbations: Defending Adversarial Attacks via Neuron Influence**

cs.CV

Final version. Accepted to IEEE Transactions on Dependable and Secure  Computing

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2112.13060v3) [paper-pdf](http://arxiv.org/pdf/2112.13060v3)

**Authors**: Ruoxi Chen, Haibo Jin, Haibin Zheng, Jinyin Chen, Zhenguang Liu

**Abstract**: The vulnerabilities of deep learning models towards adversarial attacks have attracted increasing attention, especially when models are deployed in security-critical domains. Numerous defense methods, including reactive and proactive ones, have been proposed for model robustness improvement. Reactive defenses, such as conducting transformations to remove perturbations, usually fail to handle large perturbations. The proactive defenses that involve retraining, suffer from the attack dependency and high computation cost. In this paper, we consider defense methods from the general effect of adversarial attacks that take on neurons inside the model. We introduce the concept of neuron influence, which can quantitatively measure neurons' contribution to correct classification. Then, we observe that almost all attacks fool the model by suppressing neurons with larger influence and enhancing those with smaller influence. Based on this, we propose \emph{Neuron-level Inverse Perturbation} (NIP), a novel defense against general adversarial attacks. It calculates neuron influence from benign examples and then modifies input examples by generating inverse perturbations that can in turn strengthen neurons with larger influence and weaken those with smaller influence.



## **2. Detecting Adversarial Attacks in Semantic Segmentation via Uncertainty Estimation: A Deep Analysis**

cs.CV

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10021v1) [paper-pdf](http://arxiv.org/pdf/2408.10021v1)

**Authors**: Kira Maag, Roman Resner, Asja Fischer

**Abstract**: Deep neural networks have demonstrated remarkable effectiveness across a wide range of tasks such as semantic segmentation. Nevertheless, these networks are vulnerable to adversarial attacks that add imperceptible perturbations to the input image, leading to false predictions. This vulnerability is particularly dangerous in safety-critical applications like automated driving. While adversarial examples and defense strategies are well-researched in the context of image classification, there is comparatively less research focused on semantic segmentation. Recently, we have proposed an uncertainty-based method for detecting adversarial attacks on neural networks for semantic segmentation. We observed that uncertainty, as measured by the entropy of the output distribution, behaves differently on clean versus adversely perturbed images, and we utilize this property to differentiate between the two. In this extended version of our work, we conduct a detailed analysis of uncertainty-based detection of adversarial attacks including a diverse set of adversarial attacks and various state-of-the-art neural networks. Our numerical experiments show the effectiveness of the proposed uncertainty-based detection method, which is lightweight and operates as a post-processing step, i.e., no model modifications or knowledge of the adversarial example generation process are required.



## **3. Adversarial Prompt Tuning for Vision-Language Models**

cs.CV

ECCV 2024

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2311.11261v3) [paper-pdf](http://arxiv.org/pdf/2311.11261v3)

**Authors**: Jiaming Zhang, Xingjun Ma, Xin Wang, Lingyu Qiu, Jiaqi Wang, Yu-Gang Jiang, Jitao Sang

**Abstract**: With the rapid advancement of multimodal learning, pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capacities in bridging the gap between visual and language modalities. However, these models remain vulnerable to adversarial attacks, particularly in the image modality, presenting considerable security risks. This paper introduces Adversarial Prompt Tuning (AdvPT), a novel technique to enhance the adversarial robustness of image encoders in VLMs. AdvPT innovatively leverages learnable text prompts and aligns them with adversarial image embeddings, to address the vulnerabilities inherent in VLMs without the need for extensive parameter training or modification of the model architecture. We demonstrate that AdvPT improves resistance against white-box and black-box adversarial attacks and exhibits a synergistic effect when combined with existing image-processing-based defense techniques, further boosting defensive capabilities. Comprehensive experimental analyses provide insights into adversarial prompt tuning, a novel paradigm devoted to improving resistance to adversarial images through textual input modifications, paving the way for future robust multimodal learning research. These findings open up new possibilities for enhancing the security of VLMs. Our code is available at https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning.



## **4. Segment-Anything Models Achieve Zero-shot Robustness in Autonomous Driving**

cs.CV

Accepted to IAVVC 2024

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.09839v1) [paper-pdf](http://arxiv.org/pdf/2408.09839v1)

**Authors**: Jun Yan, Pengyu Wang, Danni Wang, Weiquan Huang, Daniel Watzenig, Huilin Yin

**Abstract**: Semantic segmentation is a significant perception task in autonomous driving. It suffers from the risks of adversarial examples. In the past few years, deep learning has gradually transitioned from convolutional neural network (CNN) models with a relatively small number of parameters to foundation models with a huge number of parameters. The segment-anything model (SAM) is a generalized image segmentation framework that is capable of handling various types of images and is able to recognize and segment arbitrary objects in an image without the need to train on a specific object. It is a unified model that can handle diverse downstream tasks, including semantic segmentation, object detection, and tracking. In the task of semantic segmentation for autonomous driving, it is significant to study the zero-shot adversarial robustness of SAM. Therefore, we deliver a systematic empirical study on the robustness of SAM without additional training. Based on the experimental results, the zero-shot adversarial robustness of the SAM under the black-box corruptions and white-box adversarial attacks is acceptable, even without the need for additional training. The finding of this study is insightful in that the gigantic model parameters and huge amounts of training data lead to the phenomenon of emergence, which builds a guarantee of adversarial robustness. SAM is a vision foundation model that can be regarded as an early prototype of an artificial general intelligence (AGI) pipeline. In such a pipeline, a unified model can handle diverse tasks. Therefore, this research not only inspects the impact of vision foundation models on safe autonomous driving but also provides a perspective on developing trustworthy AGI. The code is available at: https://github.com/momo1986/robust_sam_iv.



## **5. Patch of Invisibility: Naturalistic Physical Black-Box Adversarial Attacks on Object Detectors**

cs.CV

Accepted at MLCS @ ECML-PKDD 2024

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2303.04238v5) [paper-pdf](http://arxiv.org/pdf/2303.04238v5)

**Authors**: Raz Lapid, Eylon Mizrahi, Moshe Sipper

**Abstract**: Adversarial attacks on deep-learning models have been receiving increased attention in recent years. Work in this area has mostly focused on gradient-based techniques, so-called "white-box" attacks, wherein the attacker has access to the targeted model's internal parameters; such an assumption is usually unrealistic in the real world. Some attacks additionally use the entire pixel space to fool a given model, which is neither practical nor physical (i.e., real-world). On the contrary, we propose herein a direct, black-box, gradient-free method that uses the learned image manifold of a pretrained generative adversarial network (GAN) to generate naturalistic physical adversarial patches for object detectors. To our knowledge this is the first and only method that performs black-box physical attacks directly on object-detection models, which results with a model-agnostic attack. We show that our proposed method works both digitally and physically. We compared our approach against four different black-box attacks with different configurations. Our approach outperformed all other approaches that were tested in our experiments by a large margin.



## **6. Regularization for Adversarial Robust Learning**

cs.LG

51 pages, 5 figures

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.09672v1) [paper-pdf](http://arxiv.org/pdf/2408.09672v1)

**Authors**: Jie Wang, Rui Gao, Yao Xie

**Abstract**: Despite the growing prevalence of artificial neural networks in real-world applications, their vulnerability to adversarial attacks remains to be a significant concern, which motivates us to investigate the robustness of machine learning models. While various heuristics aim to optimize the distributionally robust risk using the $\infty$-Wasserstein metric, such a notion of robustness frequently encounters computation intractability. To tackle the computational challenge, we develop a novel approach to adversarial training that integrates $\phi$-divergence regularization into the distributionally robust risk function. This regularization brings a notable improvement in computation compared with the original formulation. We develop stochastic gradient methods with biased oracles to solve this problem efficiently, achieving the near-optimal sample complexity. Moreover, we establish its regularization effects and demonstrate it is asymptotic equivalence to a regularized empirical risk minimization (ERM) framework, by considering various scaling regimes of the regularization parameter $\eta$ and robustness level $\rho$. These regimes yield gradient norm regularization, variance regularization, or a smoothed gradient norm regularization that interpolates between these extremes. We numerically validate our proposed method in supervised learning, reinforcement learning, and contextual learning and showcase its state-of-the-art performance against various adversarial attacks.



## **7. Symbiotic Game and Foundation Models for Cyber Deception Operations in Strategic Cyber Warfare**

cs.CR

40 pages, 7 figures, 2 tables

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2403.10570v2) [paper-pdf](http://arxiv.org/pdf/2403.10570v2)

**Authors**: Tao Li, Quanyan Zhu

**Abstract**: We are currently facing unprecedented cyber warfare with the rapid evolution of tactics, increasing asymmetry of intelligence, and the growing accessibility of hacking tools. In this landscape, cyber deception emerges as a critical component of our defense strategy against increasingly sophisticated attacks. This chapter aims to highlight the pivotal role of game-theoretic models and foundation models (FMs) in analyzing, designing, and implementing cyber deception tactics. Game models (GMs) serve as a foundational framework for modeling diverse adversarial interactions, allowing us to encapsulate both adversarial knowledge and domain-specific insights. Meanwhile, FMs serve as the building blocks for creating tailored machine learning models suited to given applications. By leveraging the synergy between GMs and FMs, we can advance proactive and automated cyber defense mechanisms by not only securing our networks against attacks but also enhancing their resilience against well-planned operations. This chapter discusses the games at the tactical, operational, and strategic levels of warfare, delves into the symbiotic relationship between these methodologies, and explores relevant applications where such a framework can make a substantial impact in cybersecurity. The chapter discusses the promising direction of the multi-agent neurosymbolic conjectural learning (MANSCOL), which allows the defender to predict adversarial behaviors, design adaptive defensive deception tactics, and synthesize knowledge for the operational level synthesis and adaptation. FMs serve as pivotal tools across various functions for MANSCOL, including reinforcement learning, knowledge assimilation, formation of conjectures, and contextual representation. This chapter concludes with a discussion of the challenges associated with FMs and their application in the domain of cybersecurity.



## **8. Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework**

cs.CR

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2312.00029v3) [paper-pdf](http://arxiv.org/pdf/2312.00029v3)

**Authors**: Matthew Pisano, Peter Ly, Abraham Sanders, Bingsheng Yao, Dakuo Wang, Tomek Strzalkowski, Mei Si

**Abstract**: Research into AI alignment has grown considerably since the recent introduction of increasingly capable Large Language Models (LLMs). Unfortunately, modern methods of alignment still fail to fully prevent harmful responses when models are deliberately attacked. Such vulnerabilities can lead to LLMs being manipulated into generating hazardous content: from instructions for creating dangerous materials to inciting violence or endorsing unethical behaviors. To help mitigate this issue, we introduce Bergeron: a framework designed to improve the robustness of LLMs against attacks without any additional parameter fine-tuning. Bergeron is organized into two tiers; with a secondary LLM acting as a guardian to the primary LLM. This framework better safeguards the primary model against incoming attacks while monitoring its output for any harmful content. Empirical analysis reviews that by using Bergeron to complement models with existing alignment training, we can significantly improve the robustness and safety of multiple, commonly used commercial and open-source LLMs. Specifically, we found that models integrated with Bergeron are, on average, nearly seven times more resistant to attacks compared to models without such support.



## **9. Interpreting Global Perturbation Robustness of Image Models using Axiomatic Spectral Importance Decomposition**

cs.AI

Accepted by Transactions on Machine Learning Research (TMLR 2024)

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.01139v2) [paper-pdf](http://arxiv.org/pdf/2408.01139v2)

**Authors**: Róisín Luo, James McDermott, Colm O'Riordan

**Abstract**: Perturbation robustness evaluates the vulnerabilities of models, arising from a variety of perturbations, such as data corruptions and adversarial attacks. Understanding the mechanisms of perturbation robustness is critical for global interpretability. We present a model-agnostic, global mechanistic interpretability method to interpret the perturbation robustness of image models. This research is motivated by two key aspects. First, previous global interpretability works, in tandem with robustness benchmarks, e.g. mean corruption error (mCE), are not designed to directly interpret the mechanisms of perturbation robustness within image models. Second, we notice that the spectral signal-to-noise ratios (SNR) of perturbed natural images exponentially decay over the frequency. This power-law-like decay implies that: Low-frequency signals are generally more robust than high-frequency signals -- yet high classification accuracy can not be achieved by low-frequency signals alone. By applying Shapley value theory, our method axiomatically quantifies the predictive powers of robust features and non-robust features within an information theory framework. Our method, dubbed as \textbf{I-ASIDE} (\textbf{I}mage \textbf{A}xiomatic \textbf{S}pectral \textbf{I}mportance \textbf{D}ecomposition \textbf{E}xplanation), provides a unique insight into model robustness mechanisms. We conduct extensive experiments over a variety of vision models pre-trained on ImageNet to show that \textbf{I-ASIDE} can not only \textbf{measure} the perturbation robustness but also \textbf{provide interpretations} of its mechanisms.



## **10. Enhancing Adversarial Transferability with Adversarial Weight Tuning**

cs.CR

13 pages

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09469v1) [paper-pdf](http://arxiv.org/pdf/2408.09469v1)

**Authors**: Jiahao Chen, Zhou Feng, Rui Zeng, Yuwen Pu, Chunyi Zhou, Yi Jiang, Yuyou Gan, Jinbao Li, Shouling Ji, Shouling_Ji

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial examples (AEs) that mislead the model while appearing benign to human observers. A critical concern is the transferability of AEs, which enables black-box attacks without direct access to the target model. However, many previous attacks have failed to explain the intrinsic mechanism of adversarial transferability. In this paper, we rethink the property of transferable AEs and reformalize the formulation of transferability. Building on insights from this mechanism, we analyze the generalization of AEs across models with different architectures and prove that we can find a local perturbation to mitigate the gap between surrogate and target models. We further establish the inner connections between model smoothness and flat local maxima, both of which contribute to the transferability of AEs. Further, we propose a new adversarial attack algorithm, \textbf{A}dversarial \textbf{W}eight \textbf{T}uning (AWT), which adaptively adjusts the parameters of the surrogate model using generated AEs to optimize the flat local maxima and model smoothness simultaneously, without the need for extra data. AWT is a data-free tuning method that combines gradient-based and model-based attack methods to enhance the transferability of AEs. Extensive experiments on a variety of models with different architectures on ImageNet demonstrate that AWT yields superior performance over other attacks, with an average increase of nearly 5\% and 10\% attack success rates on CNN-based and Transformer-based models, respectively, compared to state-of-the-art attacks.



## **11. WPN: An Unlearning Method Based on N-pair Contrastive Learning in Language Models**

cs.CL

ECAI 2024

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09459v1) [paper-pdf](http://arxiv.org/pdf/2408.09459v1)

**Authors**: Guitao Chen, Yunshen Wang, Hongye Sun, Guang Chen

**Abstract**: Generative language models (LMs) offer numerous advantages but may produce inappropriate or harmful outputs due to the harmful knowledge acquired during pre-training. This knowledge often manifests as undesirable correspondences, such as "harmful prompts" leading to "harmful outputs," which our research aims to mitigate through unlearning techniques.However, existing unlearning methods based on gradient ascent can significantly impair the performance of LMs. To address this issue, we propose a novel approach called Weighted Positional N-pair (WPN) Learning, which leverages position-weighted mean pooling within an n-pair contrastive learning framework. WPN is designed to modify the output distribution of LMs by eliminating specific harmful outputs (e.g., replacing toxic responses with neutral ones), thereby transforming the model's behavior from "harmful prompt-harmful output" to "harmful prompt-harmless response".Experiments on OPT and GPT-NEO LMs show that WPN effectively reduces the proportion of harmful responses, achieving a harmless rate of up to 95.8\% while maintaining stable performance on nine common benchmarks (with less than 2\% degradation on average). Moreover, we provide empirical evidence to demonstrate WPN's ability to weaken the harmful correspondences in terms of generalizability and robustness, as evaluated on out-of-distribution test sets and under adversarial attacks.



## **12. XAI-Based Detection of Adversarial Attacks on Deepfake Detectors**

cs.CR

Accepted at TMLR 2024

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2403.02955v2) [paper-pdf](http://arxiv.org/pdf/2403.02955v2)

**Authors**: Ben Pinhasov, Raz Lapid, Rony Ohayon, Moshe Sipper, Yehudit Aperstein

**Abstract**: We introduce a novel methodology for identifying adversarial attacks on deepfake detectors using eXplainable Artificial Intelligence (XAI). In an era characterized by digital advancement, deepfakes have emerged as a potent tool, creating a demand for efficient detection systems. However, these systems are frequently targeted by adversarial attacks that inhibit their performance. We address this gap, developing a defensible deepfake detector by leveraging the power of XAI. The proposed methodology uses XAI to generate interpretability maps for a given method, providing explicit visualizations of decision-making factors within the AI models. We subsequently employ a pretrained feature extractor that processes both the input image and its corresponding XAI image. The feature embeddings extracted from this process are then used for training a simple yet effective classifier. Our approach contributes not only to the detection of deepfakes but also enhances the understanding of possible adversarial attacks, pinpointing potential vulnerabilities. Furthermore, this approach does not change the performance of the deepfake detector. The paper demonstrates promising results suggesting a potential pathway for future deepfake detection mechanisms. We believe this study will serve as a valuable contribution to the community, sparking much-needed discourse on safeguarding deepfake detectors.



## **13. Adversarial Attacked Teacher for Unsupervised Domain Adaptive Object Detection**

cs.CV

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09431v1) [paper-pdf](http://arxiv.org/pdf/2408.09431v1)

**Authors**: Kaiwen Wang, Yinzhe Shen, Martin Lauer

**Abstract**: Object detectors encounter challenges in handling domain shifts. Cutting-edge domain adaptive object detection methods use the teacher-student framework and domain adversarial learning to generate domain-invariant pseudo-labels for self-training. However, the pseudo-labels generated by the teacher model tend to be biased towards the majority class and often mistakenly include overconfident false positives and underconfident false negatives. We reveal that pseudo-labels vulnerable to adversarial attacks are more likely to be low-quality. To address this, we propose a simple yet effective framework named Adversarial Attacked Teacher (AAT) to improve the quality of pseudo-labels. Specifically, we apply adversarial attacks to the teacher model, prompting it to generate adversarial pseudo-labels to correct bias, suppress overconfidence, and encourage underconfident proposals. An adaptive pseudo-label regularization is introduced to emphasize the influence of pseudo-labels with high certainty and reduce the negative impacts of uncertain predictions. Moreover, robust minority objects verified by pseudo-label regularization are oversampled to minimize dataset imbalance without introducing false positives. Extensive experiments conducted on various datasets demonstrate that AAT achieves superior performance, reaching 52.6 mAP on Clipart1k, surpassing the previous state-of-the-art by 6.7%.



## **14. Rethinking Impersonation and Dodging Attacks on Face Recognition Systems**

cs.CV

Accepted to ACM MM 2024

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2401.08903v4) [paper-pdf](http://arxiv.org/pdf/2401.08903v4)

**Authors**: Fengfan Zhou, Qianyu Zhou, Bangjie Yin, Hui Zheng, Xuequan Lu, Lizhuang Ma, Hefei Ling

**Abstract**: Face Recognition (FR) systems can be easily deceived by adversarial examples that manipulate benign face images through imperceptible perturbations. Adversarial attacks on FR encompass two types: impersonation (targeted) attacks and dodging (untargeted) attacks. Previous methods often achieve a successful impersonation attack on FR, however, it does not necessarily guarantee a successful dodging attack on FR in the black-box setting. In this paper, our key insight is that the generation of adversarial examples should perform both impersonation and dodging attacks simultaneously. To this end, we propose a novel attack method termed as Adversarial Pruning (Adv-Pruning), to fine-tune existing adversarial examples to enhance their dodging capabilities while preserving their impersonation capabilities. Adv-Pruning consists of Priming, Pruning, and Restoration stages. Concretely, we propose Adversarial Priority Quantification to measure the region-wise priority of original adversarial perturbations, identifying and releasing those with minimal impact on absolute model output variances. Then, Biased Gradient Adaptation is presented to adapt the adversarial examples to traverse the decision boundaries of both the attacker and victim by adding perturbations favoring dodging attacks on the vacated regions, preserving the prioritized features of the original perturbations while boosting dodging performance. As a result, we can maintain the impersonation capabilities of original adversarial examples while effectively enhancing dodging capabilities. Comprehensive experiments demonstrate the superiority of our method compared with state-of-the-art adversarial attack methods.



## **15. Malacopula: adversarial automatic speaker verification attacks using a neural-based generalised Hammerstein model**

eess.AS

Accepted at ASVspoof Workshop 2024

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09300v1) [paper-pdf](http://arxiv.org/pdf/2408.09300v1)

**Authors**: Massimiliano Todisco, Michele Panariello, Xin Wang, Héctor Delgado, Kong Aik Lee, Nicholas Evans

**Abstract**: We present Malacopula, a neural-based generalised Hammerstein model designed to introduce adversarial perturbations to spoofed speech utterances so that they better deceive automatic speaker verification (ASV) systems. Using non-linear processes to modify speech utterances, Malacopula enhances the effectiveness of spoofing attacks. The model comprises parallel branches of polynomial functions followed by linear time-invariant filters. The adversarial optimisation procedure acts to minimise the cosine distance between speaker embeddings extracted from spoofed and bona fide utterances. Experiments, performed using three recent ASV systems and the ASVspoof 2019 dataset, show that Malacopula increases vulnerabilities by a substantial margin. However, speech quality is reduced and attacks can be detected effectively under controlled conditions. The findings emphasise the need to identify new vulnerabilities and design defences to protect ASV systems from adversarial attacks in the wild.



## **16. PADetBench: Towards Benchmarking Physical Attacks against Object Detection**

cs.CV

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09181v1) [paper-pdf](http://arxiv.org/pdf/2408.09181v1)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Lap-Pui Chau, Shaohui Mei

**Abstract**: Physical attacks against object detection have gained increasing attention due to their significant practical implications.   However, conducting physical experiments is extremely time-consuming and labor-intensive.   Moreover, physical dynamics and cross-domain transformation are challenging to strictly regulate in the real world, leading to unaligned evaluation and comparison, severely hindering the development of physically robust models.   To accommodate these challenges, we explore utilizing realistic simulation to thoroughly and rigorously benchmark physical attacks with fairness under controlled physical dynamics and cross-domain transformation.   This resolves the problem of capturing identical adversarial images that cannot be achieved in the real world.   Our benchmark includes 20 physical attack methods, 48 object detectors, comprehensive physical dynamics, and evaluation metrics. We also provide end-to-end pipelines for dataset generation, detection, evaluation, and further analysis.   In addition, we perform 8064 groups of evaluation based on our benchmark, which includes both overall evaluation and further detailed ablation studies for controlled physical dynamics.   Through these experiments, we provide in-depth analyses of physical attack performance and physical adversarial robustness, draw valuable observations, and discuss potential directions for future research.   Codebase: https://github.com/JiaweiLian/Benchmarking_Physical_Attack



## **17. Training Verifiably Robust Agents Using Set-Based Reinforcement Learning**

cs.LG

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09112v1) [paper-pdf](http://arxiv.org/pdf/2408.09112v1)

**Authors**: Manuel Wendl, Lukas Koller, Tobias Ladner, Matthias Althoff

**Abstract**: Reinforcement learning often uses neural networks to solve complex control tasks. However, neural networks are sensitive to input perturbations, which makes their deployment in safety-critical environments challenging. This work lifts recent results from formally verifying neural networks against such disturbances to reinforcement learning in continuous state and action spaces using reachability analysis. While previous work mainly focuses on adversarial attacks for robust reinforcement learning, we train neural networks utilizing entire sets of perturbed inputs and maximize the worst-case reward. The obtained agents are verifiably more robust than agents obtained by related work, making them more applicable in safety-critical environments. This is demonstrated with an extensive empirical evaluation of four different benchmarks.



## **18. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

cs.CR

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09093v1) [paper-pdf](http://arxiv.org/pdf/2408.09093v1)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.



## **19. HookChain: A new perspective for Bypassing EDR Solutions**

cs.CR

50 pages, 23 figures, HookChain, Bypass EDR, Evading EDR, IAT Hook,  Halo's Gate

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2404.16856v3) [paper-pdf](http://arxiv.org/pdf/2404.16856v3)

**Authors**: Helvio Carvalho Junior

**Abstract**: In the current digital security ecosystem, where threats evolve rapidly and with complexity, companies developing Endpoint Detection and Response (EDR) solutions are in constant search for innovations that not only keep up but also anticipate emerging attack vectors. In this context, this article introduces the HookChain, a look from another perspective at widely known techniques, which when combined, provide an additional layer of sophisticated evasion against traditional EDR systems. Through a precise combination of IAT Hooking techniques, dynamic SSN resolution, and indirect system calls, HookChain redirects the execution flow of Windows subsystems in a way that remains invisible to the vigilant eyes of EDRs that only act on Ntdll.dll, without requiring changes to the source code of the applications and malwares involved. This work not only challenges current conventions in cybersecurity but also sheds light on a promising path for future protection strategies, leveraging the understanding that continuous evolution is key to the effectiveness of digital security. By developing and exploring the HookChain technique, this study significantly contributes to the body of knowledge in endpoint security, stimulating the development of more robust and adaptive solutions that can effectively address the ever-changing dynamics of digital threats. This work aspires to inspire deep reflection and advancement in the research and development of security technologies that are always several steps ahead of adversaries.



## **20. Ask, Attend, Attack: A Effective Decision-Based Black-Box Targeted Attack for Image-to-Text Models**

cs.AI

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08989v1) [paper-pdf](http://arxiv.org/pdf/2408.08989v1)

**Authors**: Qingyuan Zeng, Zhenzhong Wang, Yiu-ming Cheung, Min Jiang

**Abstract**: While image-to-text models have demonstrated significant advancements in various vision-language tasks, they remain susceptible to adversarial attacks. Existing white-box attacks on image-to-text models require access to the architecture, gradients, and parameters of the target model, resulting in low practicality. Although the recently proposed gray-box attacks have improved practicality, they suffer from semantic loss during the training process, which limits their targeted attack performance. To advance adversarial attacks of image-to-text models, this paper focuses on a challenging scenario: decision-based black-box targeted attacks where the attackers only have access to the final output text and aim to perform targeted attacks. Specifically, we formulate the decision-based black-box targeted attack as a large-scale optimization problem. To efficiently solve the optimization problem, a three-stage process \textit{Ask, Attend, Attack}, called \textit{AAA}, is proposed to coordinate with the solver. \textit{Ask} guides attackers to create target texts that satisfy the specific semantics. \textit{Attend} identifies the crucial regions of the image for attacking, thus reducing the search space for the subsequent \textit{Attack}. \textit{Attack} uses an evolutionary algorithm to attack the crucial regions, where the attacks are semantically related to the target texts of \textit{Ask}, thus achieving targeted attacks without semantic loss. Experimental results on transformer-based and CNN+RNN-based image-to-text models confirmed the effectiveness of our proposed \textit{AAA}.



## **21. Stochastic Bandits Robust to Adversarial Attacks**

cs.LG

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08859v1) [paper-pdf](http://arxiv.org/pdf/2408.08859v1)

**Authors**: Xuchuang Wang, Jinhang Zuo, Xutong Liu, John C. S. Lui, Mohammad Hajiesmaili

**Abstract**: This paper investigates stochastic multi-armed bandit algorithms that are robust to adversarial attacks, where an attacker can first observe the learner's action and {then} alter their reward observation. We study two cases of this model, with or without the knowledge of an attack budget $C$, defined as an upper bound of the summation of the difference between the actual and altered rewards. For both cases, we devise two types of algorithms with regret bounds having additive or multiplicative $C$ dependence terms. For the known attack budget case, we prove our algorithms achieve the regret bound of ${O}((K/\Delta)\log T + KC)$ and $\tilde{O}(\sqrt{KTC})$ for the additive and multiplicative $C$ terms, respectively, where $K$ is the number of arms, $T$ is the time horizon, $\Delta$ is the gap between the expected rewards of the optimal arm and the second-best arm, and $\tilde{O}$ hides the logarithmic factors. For the unknown case, we prove our algorithms achieve the regret bound of $\tilde{O}(\sqrt{KT} + KC^2)$ and $\tilde{O}(KC\sqrt{T})$ for the additive and multiplicative $C$ terms, respectively. In addition to these upper bound results, we provide several lower bounds showing the tightness of our bounds and the optimality of our algorithms. These results delineate an intrinsic separation between the bandits with attacks and corruption models [Lykouris et al., 2018].



## **22. Potion: Towards Poison Unlearning**

cs.LG

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2406.09173v2) [paper-pdf](http://arxiv.org/pdf/2406.09173v2)

**Authors**: Stefan Schoepf, Jack Foster, Alexandra Brintrup

**Abstract**: Adversarial attacks by malicious actors on machine learning systems, such as introducing poison triggers into training datasets, pose significant risks. The challenge in resolving such an attack arises in practice when only a subset of the poisoned data can be identified. This necessitates the development of methods to remove, i.e. unlearn, poison triggers from already trained models with only a subset of the poison data available. The requirements for this task significantly deviate from privacy-focused unlearning where all of the data to be forgotten by the model is known. Previous work has shown that the undiscovered poisoned samples lead to a failure of established unlearning methods, with only one method, Selective Synaptic Dampening (SSD), showing limited success. Even full retraining, after the removal of the identified poison, cannot address this challenge as the undiscovered poison samples lead to a reintroduction of the poison trigger in the model. Our work addresses two key challenges to advance the state of the art in poison unlearning. First, we introduce a novel outlier-resistant method, based on SSD, that significantly improves model protection and unlearning performance. Second, we introduce Poison Trigger Neutralisation (PTN) search, a fast, parallelisable, hyperparameter search that utilises the characteristic "unlearning versus model protection" trade-off to find suitable hyperparameters in settings where the forget set size is unknown and the retain set is contaminated. We benchmark our contributions using ResNet-9 on CIFAR10 and WideResNet-28x10 on CIFAR100. Experimental results show that our method heals 93.72% of poison compared to SSD with 83.41% and full retraining with 40.68%. We achieve this while also lowering the average model accuracy drop caused by unlearning from 5.68% (SSD) to 1.41% (ours).



## **23. ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Scale**

eess.AS

8 pages, ASVspoof 5 Workshop (Interspeech2024 Satellite)

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08739v1) [paper-pdf](http://arxiv.org/pdf/2408.08739v1)

**Authors**: Xin Wang, Hector Delgado, Hemlata Tak, Jee-weon Jung, Hye-jin Shim, Massimiliano Todisco, Ivan Kukanov, Xuechen Liu, Md Sahidullah, Tomi Kinnunen, Nicholas Evans, Kong Aik Lee, Junichi Yamagishi

**Abstract**: ASVspoof 5 is the fifth edition in a series of challenges that promote the study of speech spoofing and deepfake attacks, and the design of detection solutions. Compared to previous challenges, the ASVspoof 5 database is built from crowdsourced data collected from a vastly greater number of speakers in diverse acoustic conditions. Attacks, also crowdsourced, are generated and tested using surrogate detection models, while adversarial attacks are incorporated for the first time. New metrics support the evaluation of spoofing-robust automatic speaker verification (SASV) as well as stand-alone detection solutions, i.e., countermeasures without ASV. We describe the two challenge tracks, the new database, the evaluation metrics, baselines, and the evaluation platform, and present a summary of the results. Attacks significantly compromise the baseline systems, while submissions bring substantial improvements.



## **24. A Novel Buffered Federated Learning Framework for Privacy-Driven Anomaly Detection in IIoT**

cs.CR

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08722v1) [paper-pdf](http://arxiv.org/pdf/2408.08722v1)

**Authors**: Samira Kamali Poorazad, Chafika Benzaid, Tarik Taleb

**Abstract**: Industrial Internet of Things (IIoT) is highly sensitive to data privacy and cybersecurity threats. Federated Learning (FL) has emerged as a solution for preserving privacy, enabling private data to remain on local IIoT clients while cooperatively training models to detect network anomalies. However, both synchronous and asynchronous FL architectures exhibit limitations, particularly when dealing with clients with varying speeds due to data heterogeneity and resource constraints. Synchronous architecture suffers from straggler effects, while asynchronous methods encounter communication bottlenecks. Additionally, FL models are prone to adversarial inference attacks aimed at disclosing private training data. To address these challenges, we propose a Buffered FL (BFL) framework empowered by homomorphic encryption for anomaly detection in heterogeneous IIoT environments. BFL utilizes a novel weighted average time approach to mitigate both straggler effects and communication bottlenecks, ensuring fairness between clients with varying processing speeds through collaboration with a buffer-based server. The performance results, derived from two datasets, show the superiority of BFL compared to state-of-the-art FL methods, demonstrating improved accuracy and convergence speed while enhancing privacy preservation.



## **25. MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**

cs.CV

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2312.04960v3) [paper-pdf](http://arxiv.org/pdf/2312.04960v3)

**Authors**: Xiaoyun Xu, Shujian Yu, Zhuoran Liu, Stjepan Picek

**Abstract**: Vision Transformers (ViTs) achieve excellent performance in various tasks, but they are also vulnerable to adversarial attacks. Building robust ViTs is highly dependent on dedicated Adversarial Training (AT) strategies. However, current ViTs' adversarial training only employs well-established training approaches from convolutional neural network (CNN) training, where pre-training provides the basis for AT fine-tuning with the additional help of tailored data augmentations. In this paper, we take a closer look at the adversarial robustness of ViTs by providing a novel theoretical Mutual Information (MI) analysis in its autoencoder-based self-supervised pre-training. Specifically, we show that MI between the adversarial example and its latent representation in ViT-based autoencoders should be constrained by utilizing the MI bounds. Based on this finding, we propose a masked autoencoder-based pre-training method, MIMIR, that employs an MI penalty to facilitate the adversarial training of ViTs. Extensive experiments show that MIMIR outperforms state-of-the-art adversarially trained ViTs on benchmark datasets with higher natural and robust accuracy, indicating that ViTs can substantially benefit from exploiting MI. In addition, we consider two adaptive attacks by assuming that the adversary is aware of the MIMIR design, which further verifies the provided robustness.



## **26. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

cs.LG

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08685v1) [paper-pdf](http://arxiv.org/pdf/2408.08685v1)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial perturbations, especially for topology attacks, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attack. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph.



## **27. Towards Physical World Backdoor Attacks against Skeleton Action Recognition**

cs.CR

Accepted by ECCV 2024

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08671v1) [paper-pdf](http://arxiv.org/pdf/2408.08671v1)

**Authors**: Qichen Zheng, Yi Yu, Siyuan Yang, Jun Liu, Kwok-Yan Lam, Alex Kot

**Abstract**: Skeleton Action Recognition (SAR) has attracted significant interest for its efficient representation of the human skeletal structure. Despite its advancements, recent studies have raised security concerns in SAR models, particularly their vulnerability to adversarial attacks. However, such strategies are limited to digital scenarios and ineffective in physical attacks, limiting their real-world applicability. To investigate the vulnerabilities of SAR in the physical world, we introduce the Physical Skeleton Backdoor Attacks (PSBA), the first exploration of physical backdoor attacks against SAR. Considering the practicalities of physical execution, we introduce a novel trigger implantation method that integrates infrequent and imperceivable actions as triggers into the original skeleton data. By incorporating a minimal amount of this manipulated data into the training set, PSBA enables the system misclassify any skeleton sequences into the target class when the trigger action is present. We examine the resilience of PSBA in both poisoned and clean-label scenarios, demonstrating its efficacy across a range of datasets, poisoning ratios, and model architectures. Additionally, we introduce a trigger-enhancing strategy to strengthen attack performance in the clean label setting. The robustness of PSBA is tested against three distinct backdoor defenses, and the stealthiness of PSBA is evaluated using two quantitative metrics. Furthermore, by employing a Kinect V2 camera, we compile a dataset of human actions from the real world to mimic physical attack situations, with our findings confirming the effectiveness of our proposed attacks. Our project website can be found at https://qichenzheng.github.io/psba-website.



## **28. A Stealthy Wrongdoer: Feature-Oriented Reconstruction Attack against Split Learning**

cs.CR

Accepted to CVPR 2024

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2405.04115v2) [paper-pdf](http://arxiv.org/pdf/2405.04115v2)

**Authors**: Xiaoyang Xu, Mengda Yang, Wenzhe Yi, Ziang Li, Juan Wang, Hongxin Hu, Yong Zhuang, Yaxin Liu

**Abstract**: Split Learning (SL) is a distributed learning framework renowned for its privacy-preserving features and minimal computational requirements. Previous research consistently highlights the potential privacy breaches in SL systems by server adversaries reconstructing training data. However, these studies often rely on strong assumptions or compromise system utility to enhance attack performance. This paper introduces a new semi-honest Data Reconstruction Attack on SL, named Feature-Oriented Reconstruction Attack (FORA). In contrast to prior works, FORA relies on limited prior knowledge, specifically that the server utilizes auxiliary samples from the public without knowing any client's private information. This allows FORA to conduct the attack stealthily and achieve robust performance. The key vulnerability exploited by FORA is the revelation of the model representation preference in the smashed data output by victim client. FORA constructs a substitute client through feature-level transfer learning, aiming to closely mimic the victim client's representation preference. Leveraging this substitute client, the server trains the attack model to effectively reconstruct private data. Extensive experiments showcase FORA's superior performance compared to state-of-the-art methods. Furthermore, the paper systematically evaluates the proposed method's applicability across diverse settings and advanced defense strategies.



## **29. Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspective**

cs.IR

Survey paper

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2407.06992v2) [paper-pdf](http://arxiv.org/pdf/2407.06992v2)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Recent advances in neural information retrieval (IR) models have significantly enhanced their effectiveness over various IR tasks. The robustness of these models, essential for ensuring their reliability in practice, has also garnered significant attention. With a wide array of research on robust IR being proposed, we believe it is the opportune moment to consolidate the current status, glean insights from existing methodologies, and lay the groundwork for future development. We view the robustness of IR to be a multifaceted concept, emphasizing its necessity against adversarial attacks, out-of-distribution (OOD) scenarios and performance variance. With a focus on adversarial and OOD robustness, we dissect robustness solutions for dense retrieval models (DRMs) and neural ranking models (NRMs), respectively, recognizing them as pivotal components of the neural IR pipeline. We provide an in-depth discussion of existing methods, datasets, and evaluation metrics, shedding light on challenges and future directions in the era of large language models. To the best of our knowledge, this is the first comprehensive survey on the robustness of neural IR models, and we will also be giving our first tutorial presentation at SIGIR 2024 \url{https://sigir2024-robust-information-retrieval.github.io}. Along with the organization of existing work, we introduce a Benchmark for robust IR (BestIR), a heterogeneous evaluation benchmark for robust neural information retrieval, which is publicly available at \url{https://github.com/Davion-Liu/BestIR}. We hope that this study provides useful clues for future research on the robustness of IR models and helps to develop trustworthy search engines \url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.



## **30. Efficient Image-to-Image Diffusion Classifier for Adversarial Robustness**

cs.CV

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08502v1) [paper-pdf](http://arxiv.org/pdf/2408.08502v1)

**Authors**: Hefei Mei, Minjing Dong, Chang Xu

**Abstract**: Diffusion models (DMs) have demonstrated great potential in the field of adversarial robustness, where DM-based defense methods can achieve superior defense capability without adversarial training. However, they all require huge computational costs due to the usage of large-scale pre-trained DMs, making it difficult to conduct full evaluation under strong attacks and compare with traditional CNN-based methods. Simply reducing the network size and timesteps in DMs could significantly harm the image generation quality, which invalidates previous frameworks. To alleviate this issue, we redesign the diffusion framework from generating high-quality images to predicting distinguishable image labels. Specifically, we employ an image translation framework to learn many-to-one mapping from input samples to designed orthogonal image labels. Based on this framework, we introduce an efficient Image-to-Image diffusion classifier with a pruned U-Net structure and reduced diffusion timesteps. Besides the framework, we redesign the optimization objective of DMs to fit the target of image classification, where a new classification loss is incorporated in the DM-based image translation framework to distinguish the generated label from those of other classes. We conduct sufficient evaluations of the proposed classifier under various attacks on popular benchmarks. Extensive experiments show that our method achieves better adversarial robustness with fewer computational costs than DM-based and CNN-based methods. The code is available at https://github.com/hfmei/IDC.



## **31. DFT-Based Adversarial Attack Detection in MRI Brain Imaging: Enhancing Diagnostic Accuracy in Alzheimer's Case Studies**

eess.IV

10 pages, 4 figures, conference

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08489v1) [paper-pdf](http://arxiv.org/pdf/2408.08489v1)

**Authors**: Mohammad Hossein Najafi, Mohammad Morsali, Mohammadmahdi Vahediahmar, Saeed Bagheri Shouraki

**Abstract**: Recent advancements in deep learning, particularly in medical imaging, have significantly propelled the progress of healthcare systems. However, examining the robustness of medical images against adversarial attacks is crucial due to their real-world applications and profound impact on individuals' health. These attacks can result in misclassifications in disease diagnosis, potentially leading to severe consequences. Numerous studies have explored both the implementation of adversarial attacks on medical images and the development of defense mechanisms against these threats, highlighting the vulnerabilities of deep neural networks to such adversarial activities. In this study, we investigate adversarial attacks on images associated with Alzheimer's disease and propose a defensive method to counteract these attacks. Specifically, we examine adversarial attacks that employ frequency domain transformations on Alzheimer's disease images, along with other well-known adversarial attacks. Our approach utilizes a convolutional neural network (CNN)-based autoencoder architecture in conjunction with the two-dimensional Fourier transform of images for detection purposes. The simulation results demonstrate that our detection and defense mechanism effectively mitigates several adversarial attacks, thereby enhancing the robustness of deep neural networks against such vulnerabilities.



## **32. On the Impact of Uncertainty and Calibration on Likelihood-Ratio Membership Inference Attacks**

cs.IT

13 pages, 20 figures

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2402.10686v2) [paper-pdf](http://arxiv.org/pdf/2402.10686v2)

**Authors**: Meiyi Zhu, Caili Guo, Chunyan Feng, Osvaldo Simeone

**Abstract**: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the state-of-the-art likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in which an adaptive prediction set is produced as in conformal prediction. We derive bounds on the advantage of an MIA adversary with the aim of offering insights into the impact of uncertainty and calibration on the effectiveness of MIAs. Simulation results demonstrate that the derived analytical bounds predict well the effectiveness of MIAs.



## **33. A Multi-task Adversarial Attack Against Face Authentication**

cs.CV

Accepted by ACM Transactions on Multimedia Computing, Communications,  and Applications

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.08205v1) [paper-pdf](http://arxiv.org/pdf/2408.08205v1)

**Authors**: Hanrui Wang, Shuo Wang, Cunjian Chen, Massimo Tistarelli, Zhe Jin

**Abstract**: Deep-learning-based identity management systems, such as face authentication systems, are vulnerable to adversarial attacks. However, existing attacks are typically designed for single-task purposes, which means they are tailored to exploit vulnerabilities unique to the individual target rather than being adaptable for multiple users or systems. This limitation makes them unsuitable for certain attack scenarios, such as morphing, universal, transferable, and counter attacks. In this paper, we propose a multi-task adversarial attack algorithm called MTADV that are adaptable for multiple users or systems. By interpreting these scenarios as multi-task attacks, MTADV is applicable to both single- and multi-task attacks, and feasible in the white- and gray-box settings. Furthermore, MTADV is effective against various face datasets, including LFW, CelebA, and CelebA-HQ, and can work with different deep learning models, such as FaceNet, InsightFace, and CurricularFace. Importantly, MTADV retains its feasibility as a single-task attack targeting a single user/system. To the best of our knowledge, MTADV is the first adversarial attack method that can target all of the aforementioned scenarios in one algorithm.



## **34. Prefix Guidance: A Steering Wheel for Large Language Models to Defend Against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.08924v1) [paper-pdf](http://arxiv.org/pdf/2408.08924v1)

**Authors**: Jiawei Zhao, Kejiang Chen, Xiaojian Yuan, Weiming Zhang

**Abstract**: In recent years, the rapid development of large language models (LLMs) has achieved remarkable performance across various tasks. However, research indicates that LLMs are vulnerable to jailbreak attacks, where adversaries can induce the generation of harmful content through meticulously crafted prompts. This vulnerability poses significant challenges to the secure use and promotion of LLMs. Existing defense methods offer protection from different perspectives but often suffer from insufficient effectiveness or a significant impact on the model's capabilities. In this paper, we propose a plug-and-play and easy-to-deploy jailbreak defense framework, namely Prefix Guidance (PG), which guides the model to identify harmful prompts by directly setting the first few tokens of the model's output. This approach combines the model's inherent security capabilities with an external classifier to defend against jailbreak attacks. We demonstrate the effectiveness of PG across three models and five attack methods. Compared to baselines, our approach is generally more effective on average. Additionally, results on the Just-Eval benchmark further confirm PG's superiority to preserve the model's performance.



## **35. Large Language Model Sentinel: LLM Agent for Adversarial Purification**

cs.CL

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2405.20770v2) [paper-pdf](http://arxiv.org/pdf/2405.20770v2)

**Authors**: Guang Lin, Qibin Zhao

**Abstract**: Over the past two years, the use of large language models (LLMs) has advanced rapidly. While these LLMs offer considerable convenience, they also raise security concerns, as LLMs are vulnerable to adversarial attacks by some well-designed textual perturbations. In this paper, we introduce a novel defense technique named Large LAnguage MOdel Sentinel (LLAMOS), which is designed to enhance the adversarial robustness of LLMs by purifying the adversarial textual examples before feeding them into the target LLM. Our method comprises two main components: a) Agent instruction, which can simulate a new agent for adversarial defense, altering minimal characters to maintain the original meaning of the sentence while defending against attacks; b) Defense guidance, which provides strategies for modifying clean or adversarial examples to ensure effective defense and accurate outputs from the target LLMs. Remarkably, the defense agent demonstrates robust defensive capabilities even without learning from adversarial examples. Additionally, we conduct an intriguing adversarial experiment where we develop two agents, one for defense and one for attack, and engage them in mutual confrontation. During the adversarial interactions, neither agent completely beat the other. Extensive experiments on both open-source and closed-source LLMs demonstrate that our method effectively defends against adversarial attacks, thereby enhancing adversarial robustness.



## **36. Security Challenges of Complex Space Applications: An Empirical Study**

cs.CR

Presented at the ESA Security for Space Systems (3S) conference on  the 28th of May 2024

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.08061v1) [paper-pdf](http://arxiv.org/pdf/2408.08061v1)

**Authors**: Tomas Paulik

**Abstract**: Software applications in the space and defense industries have their unique characteristics: They are complex in structure, mission-critical, and often targets of state-of-the-art cyber attacks sponsored by adversary nation states. These applications have typically a high number of stakeholders in their software component supply chain, data supply chain, and user base. The aforementioned factors make such software applications potentially vulnerable to bad actors, as the widely adopted DevOps tools and practices were not designed for high-complexity and high-risk environments.   In this study, I investigate the security challenges of the development and management of complex space applications, which differentiate the process from the commonly used practices. My findings are based on interviews with five domain experts from the industry and are further supported by a comprehensive review of relevant publications.   To illustrate the dynamics of the problem, I present and discuss an actual software supply chain structure used by Thales Alenia Space, which is one of the largest suppliers of the European Space Agency. Subsequently, I discuss the four most critical security challenges identified by the interviewed experts: Verification of software artifacts, verification of the deployed application, single point of security failure, and data tampering by trusted stakeholders. Furthermore, I present best practices which could be used to overcome each of the given challenges, and whether the interviewed experts think their organization has access to the right tools to address them. Finally, I propose future research of new DevSecOps strategies, practices, and tools which would enable better methods of software integrity verification in the space and defense industries.



## **37. LiD-FL: Towards List-Decodable Federated Learning**

cs.LG

26 pages, 5 figures

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.04963v2) [paper-pdf](http://arxiv.org/pdf/2408.04963v2)

**Authors**: Hong Liu, Liren Shan, Han Bao, Ronghui You, Yuhao Yi, Jiancheng Lv

**Abstract**: Federated learning is often used in environments with many unverified participants. Therefore, federated learning under adversarial attacks receives significant attention. This paper proposes an algorithmic framework for list-decodable federated learning, where a central server maintains a list of models, with at least one guaranteed to perform well. The framework has no strict restriction on the fraction of honest workers, extending the applicability of Byzantine federated learning to the scenario with more than half adversaries. Under proper assumptions on the loss function, we prove a convergence theorem for our method. Experimental results, including image classification tasks with both convex and non-convex losses, demonstrate that the proposed algorithm can withstand the malicious majority under various attacks.



## **38. A Survey of Trojan Attacks and Defenses to Deep Neural Networks**

cs.CR

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.08920v1) [paper-pdf](http://arxiv.org/pdf/2408.08920v1)

**Authors**: Lingxin Jin, Xianyu Wen, Wei Jiang, Jinyu Zhan

**Abstract**: Deep Neural Networks (DNNs) have found extensive applications in safety-critical artificial intelligence systems, such as autonomous driving and facial recognition systems. However, recent research has revealed their susceptibility to Neural Network Trojans (NN Trojans) maliciously injected by adversaries. This vulnerability arises due to the intricate architecture and opacity of DNNs, resulting in numerous redundant neurons embedded within the models. Adversaries exploit these vulnerabilities to conceal malicious Trojans within DNNs, thereby causing erroneous outputs and posing substantial threats to the efficacy of DNN-based applications. This article presents a comprehensive survey of Trojan attacks against DNNs and the countermeasure methods employed to mitigate them. Initially, we trace the evolution of the concept from traditional Trojans to NN Trojans, highlighting the feasibility and practicality of generating NN Trojans. Subsequently, we provide an overview of notable works encompassing various attack and defense strategies, facilitating a comparative analysis of their approaches. Through these discussions, we offer constructive insights aimed at refining these techniques. In recognition of the gravity and immediacy of this subject matter, we also assess the feasibility of deploying such attacks in real-world scenarios as opposed to controlled ideal datasets. The potential real-world implications underscore the urgency of addressing this issue effectively.



## **39. Robust Active Learning (RoAL): Countering Dynamic Adversaries in Active Learning with Elastic Weight Consolidation**

cs.LG

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.07364v2) [paper-pdf](http://arxiv.org/pdf/2408.07364v2)

**Authors**: Ricky Maulana Fajri, Yulong Pei, Lu Yin, Mykola Pechenizkiy

**Abstract**: Despite significant advancements in active learning and adversarial attacks, the intersection of these two fields remains underexplored, particularly in developing robust active learning frameworks against dynamic adversarial threats. The challenge of developing robust active learning frameworks under dynamic adversarial attacks is critical, as these attacks can lead to catastrophic forgetting within the active learning cycle. This paper introduces Robust Active Learning (RoAL), a novel approach designed to address this issue by integrating Elastic Weight Consolidation (EWC) into the active learning process. Our contributions are threefold: First, we propose a new dynamic adversarial attack that poses significant threats to active learning frameworks. Second, we introduce a novel method that combines EWC with active learning to mitigate catastrophic forgetting caused by dynamic adversarial attacks. Finally, we conduct extensive experimental evaluations to demonstrate the efficacy of our approach. The results show that RoAL not only effectively counters dynamic adversarial threats but also significantly reduces the impact of catastrophic forgetting, thereby enhancing the robustness and performance of active learning systems in adversarial environments.



## **40. Enhancing Adversarial Attacks via Parameter Adaptive Adversarial Attack**

cs.LG

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2408.07733v1) [paper-pdf](http://arxiv.org/pdf/2408.07733v1)

**Authors**: Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Chenyu Zhang, Jiahao Huang, Jianlong Zhou, Fang Chen

**Abstract**: In recent times, the swift evolution of adversarial attacks has captured widespread attention, particularly concerning their transferability and other performance attributes. These techniques are primarily executed at the sample level, frequently overlooking the intrinsic parameters of models. Such neglect suggests that the perturbations introduced in adversarial samples might have the potential for further reduction. Given the essence of adversarial attacks is to impair model integrity with minimal noise on original samples, exploring avenues to maximize the utility of such perturbations is imperative. Against this backdrop, we have delved into the complexities of adversarial attack algorithms, dissecting the adversarial process into two critical phases: the Directional Supervision Process (DSP) and the Directional Optimization Process (DOP). While DSP determines the direction of updates based on the current samples and model parameters, it has been observed that existing model parameters may not always be conducive to adversarial attacks. The impact of models on adversarial efficacy is often overlooked in current research, leading to the neglect of DSP. We propose that under certain conditions, fine-tuning model parameters can significantly enhance the quality of DSP. For the first time, we propose that under certain conditions, fine-tuning model parameters can significantly improve the quality of the DSP. We provide, for the first time, rigorous mathematical definitions and proofs for these conditions, and introduce multiple methods for fine-tuning model parameters within DSP. Our extensive experiments substantiate the effectiveness of the proposed P3A method. Our code is accessible at: https://anonymous.4open.science/r/P3A-A12C/



## **41. TabularBench: Benchmarking Adversarial Robustness for Tabular Deep Learning in Real-world Use-cases**

cs.LG

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2408.07579v1) [paper-pdf](http://arxiv.org/pdf/2408.07579v1)

**Authors**: Thibault Simonetto, Salah Ghamizi, Maxime Cordy

**Abstract**: While adversarial robustness in computer vision is a mature research field, fewer researchers have tackled the evasion attacks against tabular deep learning, and even fewer investigated robustification mechanisms and reliable defenses. We hypothesize that this lag in the research on tabular adversarial attacks is in part due to the lack of standardized benchmarks. To fill this gap, we propose TabularBench, the first comprehensive benchmark of robustness of tabular deep learning classification models. We evaluated adversarial robustness with CAA, an ensemble of gradient and search attacks which was recently demonstrated as the most effective attack against a tabular model. In addition to our open benchmark (https://github.com/serval-uni-lu/tabularbench) where we welcome submissions of new models and defenses, we implement 7 robustification mechanisms inspired by state-of-the-art defenses in computer vision and propose the largest benchmark of robust tabular deep learning over 200 models across five critical scenarios in finance, healthcare and security. We curated real datasets for each use case, augmented with hundreds of thousands of realistic synthetic inputs, and trained and assessed our models with and without data augmentations. We open-source our library that provides API access to all our pre-trained robust tabular models, and the largest datasets of real and synthetic tabular inputs. Finally, we analyze the impact of various defenses on the robustness and provide actionable insights to design new defenses and robustification mechanisms.



## **42. Achieving Data Efficient Neural Networks with Hybrid Concept-based Models**

cs.LG

11 pages, 8 figures, appendix

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2408.07438v1) [paper-pdf](http://arxiv.org/pdf/2408.07438v1)

**Authors**: Tobias A. Opsahl, Vegard Antun

**Abstract**: Most datasets used for supervised machine learning consist of a single label per data point. However, in cases where more information than just the class label is available, would it be possible to train models more efficiently? We introduce two novel model architectures, which we call hybrid concept-based models, that train using both class labels and additional information in the dataset referred to as concepts. In order to thoroughly assess their performance, we introduce ConceptShapes, an open and flexible class of datasets with concept labels. We show that the hybrid concept-based models outperform standard computer vision models and previously proposed concept-based models with respect to accuracy, especially in sparse data settings. We also introduce an algorithm for performing adversarial concept attacks, where an image is perturbed in a way that does not change a concept-based model's concept predictions, but changes the class prediction. The existence of such adversarial examples raises questions about the interpretable qualities promised by concept-based models.



## **43. A Survey of Fragile Model Watermarking**

cs.CR

There will be more revisions to the wording and image additions, and  we hope to withdraw the previous version

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2406.04809v5) [paper-pdf](http://arxiv.org/pdf/2406.04809v5)

**Authors**: Zhenzhe Gao, Yu Cheng, Zhaoxia Yin

**Abstract**: Model fragile watermarking, inspired by both the field of adversarial attacks on neural networks and traditional multimedia fragile watermarking, has gradually emerged as a potent tool for detecting tampering, and has witnessed rapid development in recent years. Unlike robust watermarks, which are widely used for identifying model copyrights, fragile watermarks for models are designed to identify whether models have been subjected to unexpected alterations such as backdoors, poisoning, compression, among others. These alterations can pose unknown risks to model users, such as misidentifying stop signs as speed limit signs in classic autonomous driving scenarios. This paper provides an overview of the relevant work in the field of model fragile watermarking since its inception, categorizing them and revealing the developmental trajectory of the field, thus offering a comprehensive survey for future endeavors in model fragile watermarking.



## **44. BadMerging: Backdoor Attacks Against Model Merging**

cs.CR

To appear in ACM Conference on Computer and Communications Security  (CCS), 2024

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2408.07362v1) [paper-pdf](http://arxiv.org/pdf/2408.07362v1)

**Authors**: Jinghuai Zhang, Jianfeng Chi, Zheng Li, Kunlin Cai, Yang Zhang, Yuan Tian

**Abstract**: Fine-tuning pre-trained models for downstream tasks has led to a proliferation of open-sourced task-specific models. Recently, Model Merging (MM) has emerged as an effective approach to facilitate knowledge transfer among these independently fine-tuned models. MM directly combines multiple fine-tuned task-specific models into a merged model without additional training, and the resulting model shows enhanced capabilities in multiple tasks. Although MM provides great utility, it may come with security risks because an adversary can exploit MM to affect multiple downstream tasks. However, the security risks of MM have barely been studied. In this paper, we first find that MM, as a new learning paradigm, introduces unique challenges for existing backdoor attacks due to the merging process. To address these challenges, we introduce BadMerging, the first backdoor attack specifically designed for MM. Notably, BadMerging allows an adversary to compromise the entire merged model by contributing as few as one backdoored task-specific model. BadMerging comprises a two-stage attack mechanism and a novel feature-interpolation-based loss to enhance the robustness of embedded backdoors against the changes of different merging parameters. Considering that a merged model may incorporate tasks from different domains, BadMerging can jointly compromise the tasks provided by the adversary (on-task attack) and other contributors (off-task attack) and solve the corresponding unique challenges with novel attack designs. Extensive experiments show that BadMerging achieves remarkable attacks against various MM algorithms. Our ablation study demonstrates that the proposed attack designs can progressively contribute to the attack performance. Finally, we show that prior defense mechanisms fail to defend against our attacks, highlighting the need for more advanced defense.



## **45. Graph Agent Network: Empowering Nodes with Inference Capabilities for Adversarial Resilience**

cs.LG

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2306.06909v3) [paper-pdf](http://arxiv.org/pdf/2306.06909v3)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Guangquan Xu, Pan Zhou, Wengang Ma, Hanyuan Huang

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.



## **46. Elephants Do Not Forget: Differential Privacy with State Continuity for Privacy Budget**

cs.CR

In Proceedings of the 2024 ACM SIGSAC Conference on Computer and  Communications Security (CCS 2024)

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2401.17628v2) [paper-pdf](http://arxiv.org/pdf/2401.17628v2)

**Authors**: Jiankai Jin, Chitchanok Chuengsatiansup, Toby Murray, Benjamin I. P. Rubinstein, Yuval Yarom, Olga Ohrimenko

**Abstract**: Current implementations of differentially-private (DP) systems either lack support to track the global privacy budget consumed on a dataset, or fail to faithfully maintain the state continuity of this budget. We show that failure to maintain a privacy budget enables an adversary to mount replay, rollback and fork attacks - obtaining answers to many more queries than what a secure system would allow. As a result the attacker can reconstruct secret data that DP aims to protect - even if DP code runs in a Trusted Execution Environment (TEE). We propose ElephantDP, a system that aims to provide the same guarantees as a trusted curator in the global DP model would, albeit set in an untrusted environment. Our system relies on a state continuity module to provide protection for the privacy budget and a TEE to faithfully execute DP code and update the budget. To provide security, our protocol makes several design choices including the content of the persistent state and the order between budget updates and query answers. We prove that ElephantDP provides liveness (i.e., the protocol can restart from a correct state and respond to queries as long as the budget is not exceeded) and DP confidentiality (i.e., an attacker learns about a dataset as much as it would from interacting with a trusted curator). Our implementation and evaluation of the protocol use Intel SGX as a TEE to run the DP code and a network of TEEs to maintain state continuity. Compared to an insecure baseline, we observe 1.1-3.2$\times$ overheads and lower relative overheads for complex DP queries.



## **47. Exploiting Leakage in Password Managers via Injection Attacks**

cs.CR

Full version of the paper published in USENIX Security 2024

**SubmitDate**: 2024-08-13    [abs](http://arxiv.org/abs/2408.07054v1) [paper-pdf](http://arxiv.org/pdf/2408.07054v1)

**Authors**: Andrés Fábrega, Armin Namavari, Rachit Agarwal, Ben Nassi, Thomas Ristenpart

**Abstract**: This work explores injection attacks against password managers. In this setting, the adversary (only) controls their own application client, which they use to "inject" chosen payloads to a victim's client via, for example, sharing credentials with them. The injections are interleaved with adversarial observations of some form of protected state (such as encrypted vault exports or the network traffic received by the application servers), from which the adversary backs out confidential information. We uncover a series of general design patterns in popular password managers that lead to vulnerabilities allowing an adversary to efficiently recover passwords, URLs, usernames, and attachments. We develop general attack templates to exploit these design patterns and experimentally showcase their practical efficacy via analysis of ten distinct password manager applications. We disclosed our findings to these vendors, many of which deployed mitigations.



## **48. Maintaining Adversarial Robustness in Continuous Learning**

cs.LG

**SubmitDate**: 2024-08-13    [abs](http://arxiv.org/abs/2402.11196v2) [paper-pdf](http://arxiv.org/pdf/2402.11196v2)

**Authors**: Xiaolei Ru, Xiaowei Cao, Zijia Liu, Jack Murdoch Moore, Xin-Ya Zhang, Xia Zhu, Wenjia Wei, Gang Yan

**Abstract**: Adversarial robustness is essential for security and reliability of machine learning systems. However, adversarial robustness enhanced by defense algorithms is easily erased as the neural network's weights update to learn new tasks. To address this vulnerability, it is essential to improve the capability of neural networks in terms of robust continual learning. Specially, we propose a novel gradient projection technique that effectively stabilizes sample gradients from previous data by orthogonally projecting back-propagation gradients onto a crucial subspace before using them for weight updates. This technique can maintaining robustness by collaborating with a class of defense algorithms through sample gradient smoothing. The experimental results on four benchmarks including Split-CIFAR100 and Split-miniImageNet, demonstrate that the superiority of the proposed approach in mitigating rapidly degradation of robustness during continual learning even when facing strong adversarial attacks.



## **49. DePatch: Towards Robust Adversarial Patch for Evading Person Detectors in the Real World**

cs.CV

**SubmitDate**: 2024-08-13    [abs](http://arxiv.org/abs/2408.06625v1) [paper-pdf](http://arxiv.org/pdf/2408.06625v1)

**Authors**: Jikang Cheng, Ying Zhang, Zhongyuan Wang, Zou Qin, Chen Li

**Abstract**: Recent years have seen an increasing interest in physical adversarial attacks, which aim to craft deployable patterns for deceiving deep neural networks, especially for person detectors. However, the adversarial patterns of existing patch-based attacks heavily suffer from the self-coupling issue, where a degradation, caused by physical transformations, in any small patch segment can result in a complete adversarial dysfunction, leading to poor robustness in the complex real world. Upon this observation, we introduce the Decoupled adversarial Patch (DePatch) attack to address the self-coupling issue of adversarial patches. Specifically, we divide the adversarial patch into block-wise segments, and reduce the inter-dependency among these segments through randomly erasing out some segments during the optimization. We further introduce a border shifting operation and a progressive decoupling strategy to improve the overall attack capabilities. Extensive experiments demonstrate the superior performance of our method over other physical adversarial attacks, especially in the real world.



## **50. Fooling SHAP with Output Shuffling Attacks**

cs.LG

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06509v1) [paper-pdf](http://arxiv.org/pdf/2408.06509v1)

**Authors**: Jun Yuan, Aritra Dasgupta

**Abstract**: Explainable AI~(XAI) methods such as SHAP can help discover feature attributions in black-box models. If the method reveals a significant attribution from a ``protected feature'' (e.g., gender, race) on the model output, the model is considered unfair. However, adversarial attacks can subvert the detection of XAI methods. Previous approaches to constructing such an adversarial model require access to underlying data distribution, which may not be possible in many practical scenarios. We relax this constraint and propose a novel family of attacks, called shuffling attacks, that are data-agnostic. The proposed attack strategies can adapt any trained machine learning model to fool Shapley value-based explanations. We prove that Shapley values cannot detect shuffling attacks. However, algorithms that estimate Shapley values, such as linear SHAP and SHAP, can detect these attacks with varying degrees of effectiveness. We demonstrate the efficacy of the attack strategies by comparing the performance of linear SHAP and SHAP using real-world datasets.



