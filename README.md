# Latest Adversarial Attack Papers
**update at 2024-04-16 09:27:15**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. FaceCat: Enhancing Face Recognition Security with a Unified Generative Model Framework**

cs.CV

Under review

**SubmitDate**: 2024-04-14    [abs](http://arxiv.org/abs/2404.09193v1) [paper-pdf](http://arxiv.org/pdf/2404.09193v1)

**Authors**: Jiawei Chen, Xiao Yang, Yinpeng Dong, Hang Su, Jianteng Peng, Zhaoxia Yin

**Abstract**: Face anti-spoofing (FAS) and adversarial detection (FAD) have been regarded as critical technologies to ensure the safety of face recognition systems. As a consequence of their limited practicality and generalization, some existing methods aim to devise a framework capable of concurrently detecting both threats to address the challenge. Nevertheless, these methods still encounter challenges of insufficient generalization and suboptimal robustness, potentially owing to the inherent drawback of discriminative models. Motivated by the rich structural and detailed features of face generative models, we propose FaceCat which utilizes the face generative model as a pre-trained model to improve the performance of FAS and FAD. Specifically, FaceCat elaborately designs a hierarchical fusion mechanism to capture rich face semantic features of the generative model. These features then serve as a robust foundation for a lightweight head, designed to execute FAS and FAD tasks simultaneously. As relying solely on single-modality data often leads to suboptimal performance, we further propose a novel text-guided multi-modal alignment strategy that utilizes text prompts to enrich feature representation, thereby enhancing performance. For fair evaluations, we build a comprehensive protocol with a wide range of 28 attack types to benchmark the performance. Extensive experiments validate the effectiveness of FaceCat generalizes significantly better and obtains excellent robustness against input transformations.



## **2. Annealing Self-Distillation Rectification Improves Adversarial Training**

cs.LG

Accepted to ICLR 2024

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2305.12118v2) [paper-pdf](http://arxiv.org/pdf/2305.12118v2)

**Authors**: Yu-Yu Wu, Hung-Jui Wang, Shang-Tse Chen

**Abstract**: In standard adversarial training, models are optimized to fit one-hot labels within allowable adversarial perturbation budgets. However, the ignorance of underlying distribution shifts brought by perturbations causes the problem of robust overfitting. To address this issue and enhance adversarial robustness, we analyze the characteristics of robust models and identify that robust models tend to produce smoother and well-calibrated outputs. Based on the observation, we propose a simple yet effective method, Annealing Self-Distillation Rectification (ADR), which generates soft labels as a better guidance mechanism that accurately reflects the distribution shift under attack during adversarial training. By utilizing ADR, we can obtain rectified distributions that significantly improve model robustness without the need for pre-trained models or extensive extra computation. Moreover, our method facilitates seamless plug-and-play integration with other adversarial training techniques by replacing the hard labels in their objectives. We demonstrate the efficacy of ADR through extensive experiments and strong performances across datasets.



## **3. IRAD: Implicit Representation-driven Image Resampling against Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2310.11890v3) [paper-pdf](http://arxiv.org/pdf/2310.11890v3)

**Authors**: Yue Cao, Tianlin Li, Xiaofeng Cao, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: We introduce a novel approach to counter adversarial attacks, namely, image resampling. Image resampling transforms a discrete image into a new one, simulating the process of scene recapturing or rerendering as specified by a geometrical transformation. The underlying rationale behind our idea is that image resampling can alleviate the influence of adversarial perturbations while preserving essential semantic information, thereby conferring an inherent advantage in defending against adversarial attacks. To validate this concept, we present a comprehensive study on leveraging image resampling to defend against adversarial attacks. We have developed basic resampling methods that employ interpolation strategies and coordinate shifting magnitudes. Our analysis reveals that these basic methods can partially mitigate adversarial attacks. However, they come with apparent limitations: the accuracy of clean images noticeably decreases, while the improvement in accuracy on adversarial examples is not substantial. We propose implicit representation-driven image resampling (IRAD) to overcome these limitations. First, we construct an implicit continuous representation that enables us to represent any input image within a continuous coordinate space. Second, we introduce SampleNet, which automatically generates pixel-wise shifts for resampling in response to different inputs. Furthermore, we can extend our approach to the state-of-the-art diffusion-based method, accelerating it with fewer time steps while preserving its defense capability. Extensive experiments demonstrate that our method significantly enhances the adversarial robustness of diverse deep models against various attacks while maintaining high accuracy on clean images.



## **4. Proof-of-Learning with Incentive Security**

cs.CR

22 pages, 6 figures

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2404.09005v1) [paper-pdf](http://arxiv.org/pdf/2404.09005v1)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks to the recent work of Jia et al. [2021], and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.



## **5. Stability and Generalization in Free Adversarial Training**

cs.LG

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2404.08980v1) [paper-pdf](http://arxiv.org/pdf/2404.08980v1)

**Authors**: Xiwei Cheng, Kexin Fu, Farzan Farnia

**Abstract**: While adversarial training methods have resulted in significant improvements in the deep neural nets' robustness against norm-bounded adversarial perturbations, their generalization performance from training samples to test data has been shown to be considerably worse than standard empirical risk minimization methods. Several recent studies seek to connect the generalization behavior of adversarially trained classifiers to various gradient-based min-max optimization algorithms used for their training. In this work, we study the generalization performance of adversarial training methods using the algorithmic stability framework. Specifically, our goal is to compare the generalization performance of the vanilla adversarial training scheme fully optimizing the perturbations at every iteration vs. the free adversarial training simultaneously optimizing the norm-bounded perturbations and classifier parameters. Our proven generalization bounds indicate that the free adversarial training method could enjoy a lower generalization gap between training and test samples due to the simultaneous nature of its min-max optimization algorithm. We perform several numerical experiments to evaluate the generalization performance of vanilla, fast, and free adversarial training methods. Our empirical findings also show the improved generalization performance of the free adversarial training method and further demonstrate that the better generalization result could translate to greater robustness against black-box attack schemes. The code is available at https://github.com/Xiwei-Cheng/Stability_FreeAT.



## **6. MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers**

cs.LG

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2402.02263v3) [paper-pdf](http://arxiv.org/pdf/2402.02263v3)

**Authors**: Yatong Bai, Mo Zhou, Vishal M. Patel, Somayeh Sojoudi

**Abstract**: Adversarial robustness often comes at the cost of degraded accuracy, impeding the real-life application of robust classification models. Training-based solutions for better trade-offs are limited by incompatibilities with already-trained high-performance large models, necessitating the exploration of training-free ensemble approaches. Observing that robust models are more confident in correct predictions than in incorrect ones on clean and adversarial data alike, we speculate amplifying this "benign confidence property" can reconcile accuracy and robustness in an ensemble setting. To achieve so, we propose "MixedNUTS", a training-free method where the output logits of a robust classifier and a standard non-robust classifier are processed by nonlinear transformations with only three parameters, which are optimized through an efficient algorithm. MixedNUTS then converts the transformed logits into probabilities and mixes them as the overall output. On CIFAR-10, CIFAR-100, and ImageNet datasets, experimental results with custom strong adaptive attacks demonstrate MixedNUTS's vastly improved accuracy and near-SOTA robustness -- it boosts CIFAR-100 clean accuracy by 7.86 points, sacrificing merely 0.87 points in robust accuracy.



## **7. Adversarial Patterns: Building Robust Android Malware Classifiers**

cs.CR

survey

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2203.02121v2) [paper-pdf](http://arxiv.org/pdf/2203.02121v2)

**Authors**: Dipkamal Bhusal, Nidhi Rastogi

**Abstract**: Machine learning models are increasingly being adopted across various fields, such as medicine, business, autonomous vehicles, and cybersecurity, to analyze vast amounts of data, detect patterns, and make predictions or recommendations. In the field of cybersecurity, these models have made significant improvements in malware detection. However, despite their ability to understand complex patterns from unstructured data, these models are susceptible to adversarial attacks that perform slight modifications in malware samples, leading to misclassification from malignant to benign. Numerous defense approaches have been proposed to either detect such adversarial attacks or improve model robustness. These approaches have resulted in a multitude of attack and defense techniques and the emergence of a field known as `adversarial machine learning.' In this survey paper, we provide a comprehensive review of adversarial machine learning in the context of Android malware classifiers. Android is the most widely used operating system globally and is an easy target for malicious agents. The paper first presents an extensive background on Android malware classifiers, followed by an examination of the latest advancements in adversarial attacks and defenses. Finally, the paper provides guidelines for designing robust malware classifiers and outlines research directions for the future.



## **8. JailbreakLens: Visual Analysis of Jailbreak Attacks Against Large Language Models**

cs.CR

Submitted to VIS 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08793v1) [paper-pdf](http://arxiv.org/pdf/2404.08793v1)

**Authors**: Yingchaojie Feng, Zhizhang Chen, Zhining Kang, Sijia Wang, Minfeng Zhu, Wei Zhang, Wei Chen

**Abstract**: The proliferation of large language models (LLMs) has underscored concerns regarding their security vulnerabilities, notably against jailbreak attacks, where adversaries design jailbreak prompts to circumvent safety mechanisms for potential misuse. Addressing these concerns necessitates a comprehensive analysis of jailbreak prompts to evaluate LLMs' defensive capabilities and identify potential weaknesses. However, the complexity of evaluating jailbreak performance and understanding prompt characteristics makes this analysis laborious. We collaborate with domain experts to characterize problems and propose an LLM-assisted framework to streamline the analysis process. It provides automatic jailbreak assessment to facilitate performance evaluation and support analysis of components and keywords in prompts. Based on the framework, we design JailbreakLens, a visual analysis system that enables users to explore the jailbreak performance against the target model, conduct multi-level analysis of prompt characteristics, and refine prompt instances to verify findings. Through a case study, technical evaluations, and expert interviews, we demonstrate our system's effectiveness in helping users evaluate model security and identify model weaknesses.



## **9. What is the Solution for State-Adversarial Multi-Agent Reinforcement Learning?**

cs.AI

Accepted by Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2212.02705v5) [paper-pdf](http://arxiv.org/pdf/2212.02705v5)

**Authors**: Songyang Han, Sanbao Su, Sihong He, Shuo Han, Haizhao Yang, Shaofeng Zou, Fei Miao

**Abstract**: Various methods for Multi-Agent Reinforcement Learning (MARL) have been developed with the assumption that agents' policies are based on accurate state information. However, policies learned through Deep Reinforcement Learning (DRL) are susceptible to adversarial state perturbation attacks. In this work, we propose a State-Adversarial Markov Game (SAMG) and make the first attempt to investigate different solution concepts of MARL under state uncertainties. Our analysis shows that the commonly used solution concepts of optimal agent policy and robust Nash equilibrium do not always exist in SAMGs. To circumvent this difficulty, we consider a new solution concept called robust agent policy, where agents aim to maximize the worst-case expected state value. We prove the existence of robust agent policy for finite state and finite action SAMGs. Additionally, we propose a Robust Multi-Agent Adversarial Actor-Critic (RMA3C) algorithm to learn robust policies for MARL agents under state uncertainties. Our experiments demonstrate that our algorithm outperforms existing methods when faced with state perturbations and greatly improves the robustness of MARL policies. Our code is public on https://songyanghan.github.io/what_is_solution/.



## **10. Mayhem: Targeted Corruption of Register and Stack Variables**

cs.CR

ACM ASIACCS 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2309.02545v2) [paper-pdf](http://arxiv.org/pdf/2309.02545v2)

**Authors**: Andrew J. Adiletta, M. Caner Tol, Yarkın Doröz, Berk Sunar

**Abstract**: In the past decade, many vulnerabilities were discovered in microarchitectures which yielded attack vectors and motivated the study of countermeasures. Further, architectural and physical imperfections in DRAMs led to the discovery of Rowhammer attacks which give an adversary power to introduce bit flips in a victim's memory space. Numerous studies analyzed Rowhammer and proposed techniques to prevent it altogether or to mitigate its effects.   In this work, we push the boundary and show how Rowhammer can be further exploited to inject faults into stack variables and even register values in a victim's process. We achieve this by targeting the register value that is stored in the process's stack, which subsequently is flushed out into the memory, where it becomes vulnerable to Rowhammer. When the faulty value is restored into the register, it will end up used in subsequent iterations. The register value can be stored in the stack via latent function calls in the source or by actively triggering signal handlers. We demonstrate the power of the findings by applying the techniques to bypass SUDO and SSH authentication. We further outline how MySQL and other cryptographic libraries can be targeted with the new attack vector. There are a number of challenges this work overcomes with extensive experimentation before coming together to yield an end-to-end attack on an OpenSSL digital signature: achieving co-location with stack and register variables, with synchronization provided via a blocking window. We show that stack and registers are no longer safe from the Rowhammer attack.



## **11. On the Robustness of Language Guidance for Low-Level Vision Tasks: Findings from Depth Estimation**

cs.CV

Accepted to CVPR 2024. Project webpage:  https://agneetchatterjee.com/robustness_depth_lang/

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08540v1) [paper-pdf](http://arxiv.org/pdf/2404.08540v1)

**Authors**: Agneet Chatterjee, Tejas Gokhale, Chitta Baral, Yezhou Yang

**Abstract**: Recent advances in monocular depth estimation have been made by incorporating natural language as additional guidance. Although yielding impressive results, the impact of the language prior, particularly in terms of generalization and robustness, remains unexplored. In this paper, we address this gap by quantifying the impact of this prior and introduce methods to benchmark its effectiveness across various settings. We generate "low-level" sentences that convey object-centric, three-dimensional spatial relationships, incorporate them as additional language priors and evaluate their downstream impact on depth estimation. Our key finding is that current language-guided depth estimators perform optimally only with scene-level descriptions and counter-intuitively fare worse with low level descriptions. Despite leveraging additional data, these methods are not robust to directed adversarial attacks and decline in performance with an increase in distribution shift. Finally, to provide a foundation for future research, we identify points of failures and offer insights to better understand these shortcomings. With an increasing number of methods using language for depth estimation, our findings highlight the opportunities and pitfalls that require careful consideration for effective deployment in real-world settings



## **12. VertAttack: Taking advantage of Text Classifiers' horizontal vision**

cs.CL

14 pages, 4 figures, accepted to NAACL 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08538v1) [paper-pdf](http://arxiv.org/pdf/2404.08538v1)

**Authors**: Jonathan Rusert

**Abstract**: Text classification systems have continuously improved in performance over the years. However, nearly all current SOTA classifiers have a similar shortcoming, they process text in a horizontal manner. Vertically written words will not be recognized by a classifier. In contrast, humans are easily able to recognize and read words written both horizontally and vertically. Hence, a human adversary could write problematic words vertically and the meaning would still be preserved to other humans. We simulate such an attack, VertAttack. VertAttack identifies which words a classifier is reliant on and then rewrites those words vertically. We find that VertAttack is able to greatly drop the accuracy of 4 different transformer models on 5 datasets. For example, on the SST2 dataset, VertAttack is able to drop RoBERTa's accuracy from 94 to 13%. Furthermore, since VertAttack does not replace the word, meaning is easily preserved. We verify this via a human study and find that crowdworkers are able to correctly label 77% perturbed texts perturbed, compared to 81% of the original texts. We believe VertAttack offers a look into how humans might circumvent classifiers in the future and thus inspire a look into more robust algorithms.



## **13. Adversarially Robust Spiking Neural Networks Through Conversion**

cs.NE

Transactions on Machine Learning Research (TMLR), 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2311.09266v2) [paper-pdf](http://arxiv.org/pdf/2311.09266v2)

**Authors**: Ozan Özdenizci, Robert Legenstein

**Abstract**: Spiking neural networks (SNNs) provide an energy-efficient alternative to a variety of artificial neural network (ANN) based AI applications. As the progress in neuromorphic computing with SNNs expands their use in applications, the problem of adversarial robustness of SNNs becomes more pronounced. To the contrary of the widely explored end-to-end adversarial training based solutions, we address the limited progress in scalable robust SNN training methods by proposing an adversarially robust ANN-to-SNN conversion algorithm. Our method provides an efficient approach to embrace various computationally demanding robust learning objectives that have been proposed for ANNs. During a post-conversion robust finetuning phase, our method adversarially optimizes both layer-wise firing thresholds and synaptic connectivity weights of the SNN to maintain transferred robustness gains from the pre-trained ANN. We perform experimental evaluations in a novel setting proposed to rigorously assess the robustness of SNNs, where numerous adaptive adversarial attacks that account for the spike-based operation dynamics are considered. Results show that our approach yields a scalable state-of-the-art solution for adversarially robust deep SNNs with low-latency.



## **14. Counterfactual Explanations for Face Forgery Detection via Adversarial Removal of Artifacts**

cs.CV

Accepted to ICME2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08341v1) [paper-pdf](http://arxiv.org/pdf/2404.08341v1)

**Authors**: Yang Li, Songlin Yang, Wei Wang, Ziwen He, Bo Peng, Jing Dong

**Abstract**: Highly realistic AI generated face forgeries known as deepfakes have raised serious social concerns. Although DNN-based face forgery detection models have achieved good performance, they are vulnerable to latest generative methods that have less forgery traces and adversarial attacks. This limitation of generalization and robustness hinders the credibility of detection results and requires more explanations. In this work, we provide counterfactual explanations for face forgery detection from an artifact removal perspective. Specifically, we first invert the forgery images into the StyleGAN latent space, and then adversarially optimize their latent representations with the discrimination supervision from the target detection model. We verify the effectiveness of the proposed explanations from two aspects: (1) Counterfactual Trace Visualization: the enhanced forgery images are useful to reveal artifacts by visually contrasting the original images and two different visualization methods; (2) Transferable Adversarial Attacks: the adversarial forgery images generated by attacking the detection model are able to mislead other detection models, implying the removed artifacts are general. Extensive experiments demonstrate that our method achieves over 90% attack success rate and superior attack transferability. Compared with naive adversarial noise methods, our method adopts both generative and discriminative model priors, and optimize the latent representations in a synthesis-by-analysis way, which forces the search of counterfactual explanations on the natural face manifold. Thus, more general counterfactual traces can be found and better adversarial attack transferability can be achieved.



## **15. A Survey of Neural Network Robustness Assessment in Image Recognition**

cs.CV

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08285v1) [paper-pdf](http://arxiv.org/pdf/2404.08285v1)

**Authors**: Jie Wang, Jun Ai, Minyan Lu, Haoran Su, Dan Yu, Yutao Zhang, Junda Zhu, Jingyu Liu

**Abstract**: In recent years, there has been significant attention given to the robustness assessment of neural networks. Robustness plays a critical role in ensuring reliable operation of artificial intelligence (AI) systems in complex and uncertain environments. Deep learning's robustness problem is particularly significant, highlighted by the discovery of adversarial attacks on image classification models. Researchers have dedicated efforts to evaluate robustness in diverse perturbation conditions for image recognition tasks. Robustness assessment encompasses two main techniques: robustness verification/ certification for deliberate adversarial attacks and robustness testing for random data corruptions. In this survey, we present a detailed examination of both adversarial robustness (AR) and corruption robustness (CR) in neural network assessment. Analyzing current research papers and standards, we provide an extensive overview of robustness assessment in image recognition. Three essential aspects are analyzed: concepts, metrics, and assessment methods. We investigate the perturbation metrics and range representations used to measure the degree of perturbations on images, as well as the robustness metrics specifically for the robustness conditions of classification models. The strengths and limitations of the existing methods are also discussed, and some potential directions for future research are provided.



## **16. Struggle with Adversarial Defense? Try Diffusion**

cs.CV

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08273v1) [paper-pdf](http://arxiv.org/pdf/2404.08273v1)

**Authors**: Yujie Li, Yanbin Wang, Haitao xu, Bin Liu, Jianguo Sun, Zhenhao Guo, Wenrui Ma

**Abstract**: Adversarial attacks induce misclassification by introducing subtle perturbations. Recently, diffusion models are applied to the image classifiers to improve adversarial robustness through adversarial training or by purifying adversarial noise. However, diffusion-based adversarial training often encounters convergence challenges and high computational expenses. Additionally, diffusion-based purification inevitably causes data shift and is deemed susceptible to stronger adaptive attacks. To tackle these issues, we propose the Truth Maximization Diffusion Classifier (TMDC), a generative Bayesian classifier that builds upon pre-trained diffusion models and the Bayesian theorem. Unlike data-driven classifiers, TMDC, guided by Bayesian principles, utilizes the conditional likelihood from diffusion models to determine the class probabilities of input images, thereby insulating against the influences of data shift and the limitations of adversarial training. Moreover, to enhance TMDC's resilience against more potent adversarial attacks, we propose an optimization strategy for diffusion classifiers. This strategy involves post-training the diffusion model on perturbed datasets with ground-truth labels as conditions, guiding the diffusion model to learn the data distribution and maximizing the likelihood under the ground-truth labels. The proposed method achieves state-of-the-art performance on the CIFAR10 dataset against heavy white-box attacks and strong adaptive attacks. Specifically, TMDC achieves robust accuracies of 82.81% against $l_{\infty}$ norm-bounded perturbations and 86.05% against $l_{2}$ norm-bounded perturbations, respectively, with $\epsilon=0.05$.



## **17. Combating Advanced Persistent Threats: Challenges and Solutions**

cs.CR

This work has been accepted by IEEE NETWORK in April 2024. 9 pages, 5  figures, 1 table

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2309.09498v2) [paper-pdf](http://arxiv.org/pdf/2309.09498v2)

**Authors**: Yuntao Wang, Han Liu, Zhendong Li, Zhou Su, Jiliang Li

**Abstract**: The rise of advanced persistent threats (APTs) has marked a significant cybersecurity challenge, characterized by sophisticated orchestration, stealthy execution, extended persistence, and targeting valuable assets across diverse sectors. Provenance graph-based kernel-level auditing has emerged as a promising approach to enhance visibility and traceability within intricate network environments. However, it still faces challenges including reconstructing complex lateral attack chains, detecting dynamic evasion behaviors, and defending smart adversarial subgraphs. To bridge the research gap, this paper proposes an efficient and robust APT defense scheme leveraging provenance graphs, including a network-level distributed audit model for cost-effective lateral attack reconstruction, a trust-oriented APT evasion behavior detection strategy, and a hidden Markov model based adversarial subgraph defense approach. Through prototype implementation and extensive experiments, we validate the effectiveness of our system. Lastly, crucial open research directions are outlined in this emerging field.



## **18. Practical Region-level Attack against Segment Anything Models**

cs.CV

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08255v1) [paper-pdf](http://arxiv.org/pdf/2404.08255v1)

**Authors**: Yifan Shen, Zhengyuan Li, Gang Wang

**Abstract**: Segment Anything Models (SAM) have made significant advancements in image segmentation, allowing users to segment target portions of an image with a single click (i.e., user prompt). Given its broad applications, the robustness of SAM against adversarial attacks is a critical concern. While recent works have explored adversarial attacks against a pre-defined prompt/click, their threat model is not yet realistic: (1) they often assume the user-click position is known to the attacker (point-based attack), and (2) they often operate under a white-box setting with limited transferability. In this paper, we propose a more practical region-level attack where attackers do not need to know the precise user prompt. The attack remains effective as the user clicks on any point on the target object in the image, hiding the object from SAM. Also, by adapting a spectrum transformation method, we make the attack more transferable under a black-box setting. Both control experiments and testing against real-world SAM services confirm its effectiveness.



## **19. Navigating Quantum Security Risks in Networked Environments: A Comprehensive Study of Quantum-Safe Network Protocols**

cs.CR

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08232v1) [paper-pdf](http://arxiv.org/pdf/2404.08232v1)

**Authors**: Yaser Baseri, Vikas Chouhan, Abdelhakim Hafid

**Abstract**: The emergence of quantum computing poses a formidable security challenge to network protocols traditionally safeguarded by classical cryptographic algorithms. This paper provides an exhaustive analysis of vulnerabilities introduced by quantum computing in a diverse array of widely utilized security protocols across the layers of the TCP/IP model, including TLS, IPsec, SSH, PGP, and more. Our investigation focuses on precisely identifying vulnerabilities susceptible to exploitation by quantum adversaries at various migration stages for each protocol while also assessing the associated risks and consequences for secure communication. We delve deep into the impact of quantum computing on each protocol, emphasizing potential threats posed by quantum attacks and scrutinizing the effectiveness of post-quantum cryptographic solutions. Through carefully evaluating vulnerabilities and risks that network protocols face in the post-quantum era, this study provides invaluable insights to guide the development of appropriate countermeasures. Our findings contribute to a broader comprehension of quantum computing's influence on network security and offer practical guidance for protocol designers, implementers, and policymakers in addressing the challenges stemming from the advancement of quantum computing. This comprehensive study is a crucial step toward fortifying the security of networked environments in the quantum age.



## **20. CosalPure: Learning Concept from Group Images for Robust Co-Saliency Detection**

cs.CV

This paper is accepted by CVPR 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2403.18554v2) [paper-pdf](http://arxiv.org/pdf/2403.18554v2)

**Authors**: Jiayi Zhu, Qing Guo, Felix Juefei-Xu, Yihao Huang, Yang Liu, Geguang Pu

**Abstract**: Co-salient object detection (CoSOD) aims to identify the common and salient (usually in the foreground) regions across a given group of images. Although achieving significant progress, state-of-the-art CoSODs could be easily affected by some adversarial perturbations, leading to substantial accuracy reduction. The adversarial perturbations can mislead CoSODs but do not change the high-level semantic information (e.g., concept) of the co-salient objects. In this paper, we propose a novel robustness enhancement framework by first learning the concept of the co-salient objects based on the input group images and then leveraging this concept to purify adversarial perturbations, which are subsequently fed to CoSODs for robustness enhancement. Specifically, we propose CosalPure containing two modules, i.e., group-image concept learning and concept-guided diffusion purification. For the first module, we adopt a pre-trained text-to-image diffusion model to learn the concept of co-salient objects within group images where the learned concept is robust to adversarial examples. For the second module, we map the adversarial image to the latent space and then perform diffusion generation by embedding the learned concept into the noise prediction function as an extra condition. Our method can effectively alleviate the influence of the SOTA adversarial attack containing different adversarial patterns, including exposure and noise. The extensive results demonstrate that our method could enhance the robustness of CoSODs significantly.



## **21. Systematically Assessing the Security Risks of AI/ML-enabled Connected Healthcare Systems**

cs.CR

13 pages, 5 figures, 3 tables

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2401.17136v2) [paper-pdf](http://arxiv.org/pdf/2401.17136v2)

**Authors**: Mohammed Elnawawy, Mohammadreza Hallajiyan, Gargi Mitra, Shahrear Iqbal, Karthik Pattabiraman

**Abstract**: The adoption of machine-learning-enabled systems in the healthcare domain is on the rise. While the use of ML in healthcare has several benefits, it also expands the threat surface of medical systems. We show that the use of ML in medical systems, particularly connected systems that involve interfacing the ML engine with multiple peripheral devices, has security risks that might cause life-threatening damage to a patient's health in case of adversarial interventions. These new risks arise due to security vulnerabilities in the peripheral devices and communication channels. We present a case study where we demonstrate an attack on an ML-enabled blood glucose monitoring system by introducing adversarial data points during inference. We show that an adversary can achieve this by exploiting a known vulnerability in the Bluetooth communication channel connecting the glucose meter with the ML-enabled app. We further show that state-of-the-art risk assessment techniques are not adequate for identifying and assessing these new risks. Our study highlights the need for novel risk analysis methods for analyzing the security of AI-enabled connected health devices.



## **22. Eliminating Catastrophic Overfitting Via Abnormal Adversarial Examples Regularization**

cs.LG

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.08154v1) [paper-pdf](http://arxiv.org/pdf/2404.08154v1)

**Authors**: Runqi Lin, Chaojian Yu, Tongliang Liu

**Abstract**: Single-step adversarial training (SSAT) has demonstrated the potential to achieve both efficiency and robustness. However, SSAT suffers from catastrophic overfitting (CO), a phenomenon that leads to a severely distorted classifier, making it vulnerable to multi-step adversarial attacks. In this work, we observe that some adversarial examples generated on the SSAT-trained network exhibit anomalous behaviour, that is, although these training samples are generated by the inner maximization process, their associated loss decreases instead, which we named abnormal adversarial examples (AAEs). Upon further analysis, we discover a close relationship between AAEs and classifier distortion, as both the number and outputs of AAEs undergo a significant variation with the onset of CO. Given this observation, we re-examine the SSAT process and uncover that before the occurrence of CO, the classifier already displayed a slight distortion, indicated by the presence of few AAEs. Furthermore, the classifier directly optimizing these AAEs will accelerate its distortion, and correspondingly, the variation of AAEs will sharply increase as a result. In such a vicious circle, the classifier rapidly becomes highly distorted and manifests as CO within a few iterations. These observations motivate us to eliminate CO by hindering the generation of AAEs. Specifically, we design a novel method, termed Abnormal Adversarial Examples Regularization (AAER), which explicitly regularizes the variation of AAEs to hinder the classifier from becoming distorted. Extensive experiments demonstrate that our method can effectively eliminate CO and further boost adversarial robustness with negligible additional computational overhead.



## **23. Chapter: Vulnerability of Quantum Information Systems to Collective Manipulation**

quant-ph

This is an earlier version of a final document that appears in  IntechOpen. Comments welcome to neiljohnson@gwu.edu, 20 Pages

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/1901.08873v2) [paper-pdf](http://arxiv.org/pdf/1901.08873v2)

**Authors**: Fernando J. Gómez-Ruiz, Ferney J. Rodríguez, Luis Quiroga, Neil F. Johnson

**Abstract**: The highly specialist terms `quantum computing' and `quantum information', together with the broader term `quantum technologies', now appear regularly in the mainstream media. While this is undoubtedly highly exciting for physicists and investors alike, a key question for society concerns such systems' vulnerabilities -- and in particular, their vulnerability to collective manipulation. Here we present and discuss a new form of vulnerability in such systems, that we have identified based on detailed many-body quantum mechanical calculations. The impact of this new vulnerability is that groups of adversaries can maximally disrupt these systems' global quantum state which will then jeopardize their quantum functionality. It will be almost impossible to detect these attacks since they do not change the Hamiltonian and the purity remains the same; they do not entail any real-time communication between the attackers; and they can last less than a second. We also argue that there can be an implicit amplification of such attacks because of the statistical character of modern non-state actor groups. A countermeasure could be to embed future quantum technologies within redundant classical networks. We purposely structure the discussion in this chapter so that the first sections are self-contained and can be read by non-specialists.



## **24. Fooling Contrastive Language-Image Pre-trained Models with CLIPMasterPrints**

cs.CV

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2307.03798v2) [paper-pdf](http://arxiv.org/pdf/2307.03798v2)

**Authors**: Matthias Freiberger, Peter Kun, Christian Igel, Anders Sundnes Løvlie, Sebastian Risi

**Abstract**: Models leveraging both visual and textual data such as Contrastive Language-Image Pre-training (CLIP), are the backbone of many recent advances in artificial intelligence. In this work, we show that despite their versatility, such models are vulnerable to what we refer to as fooling master images. Fooling master images are capable of maximizing the confidence score of a CLIP model for a significant number of widely varying prompts, while being either unrecognizable or unrelated to the attacked prompts for humans. The existence of such images is problematic as it could be used by bad actors to maliciously interfere with CLIP-trained image retrieval models in production with comparably small effort as a single image can attack many different prompts. We demonstrate how fooling master images for CLIP (CLIPMasterPrints) can be mined using stochastic gradient descent, projected gradient descent, or blackbox optimization. Contrary to many common adversarial attacks, the blackbox optimization approach allows us to mine CLIPMasterPrints even when the weights of the model are not accessible. We investigate the properties of the mined images, and find that images trained on a small number of image captions generalize to a much larger number of semantically related captions. We evaluate possible mitigation strategies, where we increase the robustness of the model and introduce an approach to automatically detect CLIPMasterPrints to sanitize the input of vulnerable models. Finally, we find that vulnerability to CLIPMasterPrints is related to a modality gap in contrastive pre-trained multi-modal networks. Code available at https://github.com/matfrei/CLIPMasterPrints.



## **25. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

cs.CL

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07921v1) [paper-pdf](http://arxiv.org/pdf/2404.07921v1)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.



## **26. A Measurement of Genuine Tor Traces for Realistic Website Fingerprinting**

cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07892v1) [paper-pdf](http://arxiv.org/pdf/2404.07892v1)

**Authors**: Rob Jansen, Ryan Wails, Aaron Johnson

**Abstract**: Website fingerprinting (WF) is a dangerous attack on web privacy because it enables an adversary to predict the website a user is visiting, despite the use of encryption, VPNs, or anonymizing networks such as Tor. Previous WF work almost exclusively uses synthetic datasets to evaluate the performance and estimate the feasibility of WF attacks despite evidence that synthetic data misrepresents the real world. In this paper we present GTT23, the first WF dataset of genuine Tor traces, which we obtain through a large-scale measurement of the Tor network. GTT23 represents real Tor user behavior better than any existing WF dataset, is larger than any existing WF dataset by at least an order of magnitude, and will help ground the future study of realistic WF attacks and defenses. In a detailed evaluation, we survey 25 WF datasets published over the last 15 years and compare their characteristics to those of GTT23. We discover common deficiencies of synthetic datasets that make them inferior to GTT23 for drawing meaningful conclusions about the effectiveness of WF attacks directed at real Tor users. We have made GTT23 available to promote reproducible research and to help inspire new directions for future work.



## **27. Multi-Robot Target Tracking with Sensing and Communication Danger Zones**

cs.RO

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07880v1) [paper-pdf](http://arxiv.org/pdf/2404.07880v1)

**Authors**: Jiazhen Li, Peihan Li, Yuwei Wu, Gaurav S. Sukhatme, Vijay Kumar, Lifeng Zhou

**Abstract**: Multi-robot target tracking finds extensive applications in different scenarios, such as environmental surveillance and wildfire management, which require the robustness of the practical deployment of multi-robot systems in uncertain and dangerous environments. Traditional approaches often focus on the performance of tracking accuracy with no modeling and assumption of the environments, neglecting potential environmental hazards which result in system failures in real-world deployments. To address this challenge, we investigate multi-robot target tracking in the adversarial environment considering sensing and communication attacks with uncertainty. We design specific strategies to avoid different danger zones and proposed a multi-agent tracking framework under the perilous environment. We approximate the probabilistic constraints and formulate practical optimization strategies to address computational challenges efficiently. We evaluate the performance of our proposed methods in simulations to demonstrate the ability of robots to adjust their risk-aware behaviors under different levels of environmental uncertainty and risk confidence. The proposed method is further validated via real-world robot experiments where a team of drones successfully track dynamic ground robots while being risk-aware of the sensing and/or communication danger zones.



## **28. LeapFrog: The Rowhammer Instruction Skip Attack**

cs.CR

Accepted at Hardware.io 2024

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07878v1) [paper-pdf](http://arxiv.org/pdf/2404.07878v1)

**Authors**: Andrew Adiletta, Caner Tol, Berk Sunar

**Abstract**: Since its inception, Rowhammer exploits have rapidly evolved into increasingly sophisticated threats not only compromising data integrity but also the control flow integrity of victim processes. Nevertheless, it remains a challenge for an attacker to identify vulnerable targets (i.e., Rowhammer gadgets), understand the outcome of the attempted fault, and formulate an attack that yields useful results.   In this paper, we present a new type of Rowhammer gadget, called a LeapFrog gadget, which, when present in the victim code, allows an adversary to subvert code execution to bypass a critical piece of code (e.g., authentication check logic, encryption rounds, padding in security protocols). The Leapfrog gadget manifests when the victim code stores the Program Counter (PC) value in the user or kernel stack (e.g., a return address during a function call) which, when tampered with, re-positions the return address to a location that bypasses a security-critical code pattern.   This research also presents a systematic process to identify Leapfrog gadgets. This methodology enables the automated detection of susceptible targets and the determination of optimal attack parameters. We first showcase this new attack vector through a practical demonstration on a TLS handshake client/server scenario, successfully inducing an instruction skip in a client application. We then demonstrate the attack on real-world code found in the wild, implementing an attack on OpenSSL.   Our findings extend the impact of Rowhammer attacks on control flow and contribute to the development of more robust defenses against these increasingly sophisticated threats.



## **29. Pilot Spoofing Attack on the Downlink of Cell-Free Massive MIMO: From the Perspective of Adversaries**

cs.IT

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2403.04435v2) [paper-pdf](http://arxiv.org/pdf/2403.04435v2)

**Authors**: Weiyang Xu, Ruiguang Wang, Yuan Zhang, Hien Quoc Ngo, Wei Xiang

**Abstract**: The channel hardening effect is less pronounced in the cell-free massive multiple-input multiple-output (mMIMO) system compared to its cellular counterpart, making it necessary to estimate the downlink effective channel gains to ensure decent performance. However, the downlink training inadvertently creates an opportunity for adversarial nodes to launch pilot spoofing attacks (PSAs). First, we demonstrate that adversarial distributed access points (APs) can severely degrade the achievable downlink rate. They achieve this by estimating their channels to users in the uplink training phase and then precoding and sending the same pilot sequences as those used by legitimate APs during the downlink training phase. Then, the impact of the downlink PSA is investigated by rigorously deriving a closed-form expression of the per-user achievable downlink rate. By employing the min-max criterion to optimize the power allocation coefficients, the maximum per-user achievable rate of downlink transmission is minimized from the perspective of adversarial APs. As an alternative to the downlink PSA, adversarial APs may opt to precode random interference during the downlink data transmission phase in order to disrupt legitimate communications. In this scenario, the achievable downlink rate is derived, and then power optimization algorithms are also developed. We present numerical results to showcase the detrimental impact of the downlink PSA and compare the effects of these two types of attacks.



## **30. Poisoning Prevention in Federated Learning and Differential Privacy via Stateful Proofs of Execution**

cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.06721v2) [paper-pdf](http://arxiv.org/pdf/2404.06721v2)

**Authors**: Norrathep Rattanavipanon, Ivan De Oliveira Nunes

**Abstract**: The rise in IoT-driven distributed data analytics, coupled with increasing privacy concerns, has led to a demand for effective privacy-preserving and federated data collection/model training mechanisms. In response, approaches such as Federated Learning (FL) and Local Differential Privacy (LDP) have been proposed and attracted much attention over the past few years. However, they still share the common limitation of being vulnerable to poisoning attacks wherein adversaries compromising edge devices feed forged (a.k.a. poisoned) data to aggregation back-ends, undermining the integrity of FL/LDP results.   In this work, we propose a system-level approach to remedy this issue based on a novel security notion of Proofs of Stateful Execution (PoSX) for IoT/embedded devices' software. To realize the PoSX concept, we design SLAPP: a System-Level Approach for Poisoning Prevention. SLAPP leverages commodity security features of embedded devices - in particular ARM TrustZoneM security extensions - to verifiably bind raw sensed data to their correct usage as part of FL/LDP edge device routines. As a consequence, it offers robust security guarantees against poisoning. Our evaluation, based on real-world prototypes featuring multiple cryptographic primitives and data collection schemes, showcases SLAPP's security and low overhead.



## **31. Enhancing Network Intrusion Detection Performance using Generative Adversarial Networks**

cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07464v1) [paper-pdf](http://arxiv.org/pdf/2404.07464v1)

**Authors**: Xinxing Zhao, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: Network intrusion detection systems (NIDS) play a pivotal role in safeguarding critical digital infrastructures against cyber threats. Machine learning-based detection models applied in NIDS are prevalent today. However, the effectiveness of these machine learning-based models is often limited by the evolving and sophisticated nature of intrusion techniques as well as the lack of diverse and updated training samples. In this research, a novel approach for enhancing the performance of an NIDS through the integration of Generative Adversarial Networks (GANs) is proposed. By harnessing the power of GANs in generating synthetic network traffic data that closely mimics real-world network behavior, we address a key challenge associated with NIDS training datasets, which is the data scarcity. Three distinct GAN models (Vanilla GAN, Wasserstein GAN and Conditional Tabular GAN) are implemented in this work to generate authentic network traffic patterns specifically tailored to represent the anomalous activity. We demonstrate how this synthetic data resampling technique can significantly improve the performance of the NIDS model for detecting such activity. By conducting comprehensive experiments using the CIC-IDS2017 benchmark dataset, augmented with GAN-generated data, we offer empirical evidence that shows the effectiveness of our proposed approach. Our findings show that the integration of GANs into NIDS can lead to enhancements in intrusion detection performance for attacks with limited training data, making it a promising avenue for bolstering the cybersecurity posture of organizations in an increasingly interconnected and vulnerable digital landscape.



## **32. Privacy preserving layer partitioning for Deep Neural Network models**

cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07437v1) [paper-pdf](http://arxiv.org/pdf/2404.07437v1)

**Authors**: Kishore Rajasekar, Randolph Loh, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: MLaaS (Machine Learning as a Service) has become popular in the cloud computing domain, allowing users to leverage cloud resources for running private inference of ML models on their data. However, ensuring user input privacy and secure inference execution is essential. One of the approaches to protect data privacy and integrity is to use Trusted Execution Environments (TEEs) by enabling execution of programs in secure hardware enclave. Using TEEs can introduce significant performance overhead due to the additional layers of encryption, decryption, security and integrity checks. This can lead to slower inference times compared to running on unprotected hardware. In our work, we enhance the runtime performance of ML models by introducing layer partitioning technique and offloading computations to GPU. The technique comprises two distinct partitions: one executed within the TEE, and the other carried out using a GPU accelerator. Layer partitioning exposes intermediate feature maps in the clear which can lead to reconstruction attacks to recover the input. We conduct experiments to demonstrate the effectiveness of our approach in protecting against input reconstruction attacks developed using trained conditional Generative Adversarial Network(c-GAN). The evaluation is performed on widely used models such as VGG-16, ResNet-50, and EfficientNetB0, using two datasets: ImageNet for Image classification and TON IoT dataset for cybersecurity attack detection.



## **33. Multi-granular Adversarial Attacks against Black-box Neural Ranking Models**

cs.IR

Accepted by SIGIR2024

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.01574v2) [paper-pdf](http://arxiv.org/pdf/2404.01574v2)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Adversarial ranking attacks have gained increasing attention due to their success in probing vulnerabilities, and, hence, enhancing the robustness, of neural ranking models. Conventional attack methods employ perturbations at a single granularity, e.g., word or sentence level, to target documents. However, limiting perturbations to a single level of granularity may reduce the flexibility of adversarial examples, thereby diminishing the potential threat of the attack. Therefore, we focus on generating high-quality adversarial examples by incorporating multi-granular perturbations. Achieving this objective involves tackling a combinatorial explosion problem, which requires identifying an optimal combination of perturbations across all possible levels of granularity, positions, and textual pieces. To address this challenge, we transform the multi-granular adversarial attack into a sequential decision-making process, where perturbations in the next attack step build on the perturbed document in the current attack step. Since the attack process can only access the final state without direct intermediate signals, we use reinforcement learning to perform multi-granular attacks. During the reinforcement learning process, two agents work cooperatively to identify multi-granular vulnerabilities as attack targets and organize perturbation candidates into a final perturbation sequence. Experimental results show that our attack method surpasses prevailing baselines in both attack effectiveness and imperceptibility.



## **34. Incremental Randomized Smoothing Certification**

cs.LG

ICLR 2024

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2305.19521v2) [paper-pdf](http://arxiv.org/pdf/2305.19521v2)

**Authors**: Shubham Ugare, Tarun Suresh, Debangshu Banerjee, Gagandeep Singh, Sasa Misailovic

**Abstract**: Randomized smoothing-based certification is an effective approach for obtaining robustness certificates of deep neural networks (DNNs) against adversarial attacks. This method constructs a smoothed DNN model and certifies its robustness through statistical sampling, but it is computationally expensive, especially when certifying with a large number of samples. Furthermore, when the smoothed model is modified (e.g., quantized or pruned), certification guarantees may not hold for the modified DNN, and recertifying from scratch can be prohibitively expensive.   We present the first approach for incremental robustness certification for randomized smoothing, IRS. We show how to reuse the certification guarantees for the original smoothed model to certify an approximated model with very few samples. IRS significantly reduces the computational cost of certifying modified DNNs while maintaining strong robustness guarantees. We experimentally demonstrate the effectiveness of our approach, showing up to 3x certification speedup over the certification that applies randomized smoothing of the approximate model from scratch.



## **35. Indoor Location Fingerprinting Privacy: A Comprehensive Survey**

cs.CR

Submitted to ACM Computing Surveys

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.07345v1) [paper-pdf](http://arxiv.org/pdf/2404.07345v1)

**Authors**: Amir Fathalizadeh, Vahideh Moghtadaiee, Mina Alishahi

**Abstract**: The pervasive integration of Indoor Positioning Systems (IPS) arises from the limitations of Global Navigation Satellite Systems (GNSS) in indoor environments, leading to the widespread adoption of Location-Based Services (LBS). Specifically, indoor location fingerprinting employs diverse signal fingerprints from user devices, enabling precise location identification by Location Service Providers (LSP). Despite its broad applications across various domains, indoor location fingerprinting introduces a notable privacy risk, as both LSP and potential adversaries inherently have access to this sensitive information, compromising users' privacy. Consequently, concerns regarding privacy vulnerabilities in this context necessitate a focused exploration of privacy-preserving mechanisms. In response to these concerns, this survey presents a comprehensive review of Privacy-Preserving Mechanisms in Indoor Location Fingerprinting (ILFPPM) based on cryptographic, anonymization, differential privacy (DP), and federated learning (FL) techniques. We also propose a distinctive and novel grouping of privacy vulnerabilities, adversary and attack models, and available evaluation metrics specific to indoor location fingerprinting systems. Given the identified limitations and research gaps in this survey, we highlight numerous prospective opportunities for future investigation, aiming to motivate researchers interested in advancing this field. This survey serves as a valuable reference for researchers and provides a clear overview for those beyond this specific research domain.



## **36. Towards a Game-theoretic Understanding of Explanation-based Membership Inference Attacks**

cs.AI

arXiv admin note: text overlap with arXiv:2202.02659

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.07139v1) [paper-pdf](http://arxiv.org/pdf/2404.07139v1)

**Authors**: Kavita Kumari, Murtuza Jadliwala, Sumit Kumar Jha, Anindya Maiti

**Abstract**: Model explanations improve the transparency of black-box machine learning (ML) models and their decisions; however, they can also be exploited to carry out privacy threats such as membership inference attacks (MIA). Existing works have only analyzed MIA in a single "what if" interaction scenario between an adversary and the target ML model; thus, it does not discern the factors impacting the capabilities of an adversary in launching MIA in repeated interaction settings. Additionally, these works rely on assumptions about the adversary's knowledge of the target model's structure and, thus, do not guarantee the optimality of the predefined threshold required to distinguish the members from non-members. In this paper, we delve into the domain of explanation-based threshold attacks, where the adversary endeavors to carry out MIA attacks by leveraging the variance of explanations through iterative interactions with the system comprising of the target ML model and its corresponding explanation method. We model such interactions by employing a continuous-time stochastic signaling game framework. In our framework, an adversary plays a stopping game, interacting with the system (having imperfect information about the type of an adversary, i.e., honest or malicious) to obtain explanation variance information and computing an optimal threshold to determine the membership of a datapoint accurately. First, we propose a sound mathematical formulation to prove that such an optimal threshold exists, which can be used to launch MIA. Then, we characterize the conditions under which a unique Markov perfect equilibrium (or steady state) exists in this dynamic system. By means of a comprehensive set of simulations of the proposed game model, we assess different factors that can impact the capability of an adversary to launch MIA in such repeated interaction settings.



## **37. Adversarial purification for no-reference image-quality metrics: applicability study and new methods**

cs.CV

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.06957v1) [paper-pdf](http://arxiv.org/pdf/2404.06957v1)

**Authors**: Aleksandr Gushchin, Anna Chistyakova, Vladislav Minashkin, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: Recently, the area of adversarial attacks on image quality metrics has begun to be explored, whereas the area of defences remains under-researched. In this study, we aim to cover that case and check the transferability of adversarial purification defences from image classifiers to IQA methods. In this paper, we apply several widespread attacks on IQA models and examine the success of the defences against them. The purification methodologies covered different preprocessing techniques, including geometrical transformations, compression, denoising, and modern neural network-based methods. Also, we address the challenge of assessing the efficacy of a defensive methodology by proposing ways to estimate output visual quality and the success of neutralizing attacks. Defences were tested against attack on three IQA metrics -- Linearity, MetaIQA and SPAQ. The code for attacks and defences is available at: (link is hidden for a blind review).



## **38. Simpler becomes Harder: Do LLMs Exhibit a Coherent Behavior on Simplified Corpora?**

cs.CL

Published at DeTermIt! Workshop at LREC-COLING 2024

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.06838v1) [paper-pdf](http://arxiv.org/pdf/2404.06838v1)

**Authors**: Miriam Anschütz, Edoardo Mosca, Georg Groh

**Abstract**: Text simplification seeks to improve readability while retaining the original content and meaning. Our study investigates whether pre-trained classifiers also maintain such coherence by comparing their predictions on both original and simplified inputs. We conduct experiments using 11 pre-trained models, including BERT and OpenAI's GPT 3.5, across six datasets spanning three languages. Additionally, we conduct a detailed analysis of the correlation between prediction change rates and simplification types/strengths. Our findings reveal alarming inconsistencies across all languages and models. If not promptly addressed, simplified inputs can be easily exploited to craft zero-iteration model-agnostic adversarial attacks with success rates of up to 50%



## **39. Logit Calibration and Feature Contrast for Robust Federated Learning on Non-IID Data**

cs.LG

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.06776v1) [paper-pdf](http://arxiv.org/pdf/2404.06776v1)

**Authors**: Yu Qiao, Chaoning Zhang, Apurba Adhikary, Choong Seon Hong

**Abstract**: Federated learning (FL) is a privacy-preserving distributed framework for collaborative model training on devices in edge networks. However, challenges arise due to vulnerability to adversarial examples (AEs) and the non-independent and identically distributed (non-IID) nature of data distribution among devices, hindering the deployment of adversarially robust and accurate learning models at the edge. While adversarial training (AT) is commonly acknowledged as an effective defense strategy against adversarial attacks in centralized training, we shed light on the adverse effects of directly applying AT in FL that can severely compromise accuracy, especially in non-IID challenges. Given this limitation, this paper proposes FatCC, which incorporates local logit \underline{C}alibration and global feature \underline{C}ontrast into the vanilla federated adversarial training (\underline{FAT}) process from both logit and feature perspectives. This approach can effectively enhance the federated system's robust accuracy (RA) and clean accuracy (CA). First, we propose logit calibration, where the logits are calibrated during local adversarial updates, thereby improving adversarial robustness. Second, FatCC introduces feature contrast, which involves a global alignment term that aligns each local representation with unbiased global features, thus further enhancing robustness and accuracy in federated adversarial environments. Extensive experiments across multiple datasets demonstrate that FatCC achieves comparable or superior performance gains in both CA and RA compared to other baselines.



## **40. Towards Building a Robust Toxicity Predictor**

cs.CL

ACL 2023 /

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.08690v1) [paper-pdf](http://arxiv.org/pdf/2404.08690v1)

**Authors**: Dmitriy Bespalov, Sourav Bhabesh, Yi Xiang, Liutong Zhou, Yanjun Qi

**Abstract**: Recent NLP literature pays little attention to the robustness of toxicity language predictors, while these systems are most likely to be used in adversarial contexts. This paper presents a novel adversarial attack, \texttt{ToxicTrap}, introducing small word-level perturbations to fool SOTA text classifiers to predict toxic text samples as benign. ToxicTrap exploits greedy based search strategies to enable fast and effective generation of toxic adversarial examples. Two novel goal function designs allow ToxicTrap to identify weaknesses in both multiclass and multilabel toxic language detectors. Our empirical results show that SOTA toxicity text classifiers are indeed vulnerable to the proposed attacks, attaining over 98\% attack success rates in multilabel cases. We also show how a vanilla adversarial training and its improved version can help increase robustness of a toxicity detector even against unseen attacks.



## **41. False Claims against Model Ownership Resolution**

cs.CR

13pages,3 figures. To appear in the 33rd USENIX Security Symposium  (USENIX Security '24)

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2304.06607v7) [paper-pdf](http://arxiv.org/pdf/2304.06607v7)

**Authors**: Jian Liu, Rui Zhang, Sebastian Szyller, Kui Ren, N. Asokan

**Abstract**: Deep neural network (DNN) models are valuable intellectual property of model owners, constituting a competitive advantage. Therefore, it is crucial to develop techniques to protect against model theft. Model ownership resolution (MOR) is a class of techniques that can deter model theft. A MOR scheme enables an accuser to assert an ownership claim for a suspect model by presenting evidence, such as a watermark or fingerprint, to show that the suspect model was stolen or derived from a source model owned by the accuser. Most of the existing MOR schemes prioritize robustness against malicious suspects, ensuring that the accuser will win if the suspect model is indeed a stolen model.   In this paper, we show that common MOR schemes in the literature are vulnerable to a different, equally important but insufficiently explored, robustness concern: a malicious accuser. We show how malicious accusers can successfully make false claims against independent suspect models that were not stolen. Our core idea is that a malicious accuser can deviate (without detection) from the specified MOR process by finding (transferable) adversarial examples that successfully serve as evidence against independent suspect models. To this end, we first generalize the procedures of common MOR schemes and show that, under this generalization, defending against false claims is as challenging as preventing (transferable) adversarial examples. Via systematic empirical evaluation, we show that our false claim attacks always succeed in the MOR schemes that follow our generalization, including in a real-world model: Amazon's Rekognition API.



## **42. Sandwich attack: Multi-language Mixture Adaptive Attack on LLMs**

cs.CR

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.07242v1) [paper-pdf](http://arxiv.org/pdf/2404.07242v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan

**Abstract**: Large Language Models (LLMs) are increasingly being developed and applied, but their widespread use faces challenges. These include aligning LLMs' responses with human values to prevent harmful outputs, which is addressed through safety training methods. Even so, bad actors and malicious users have succeeded in attempts to manipulate the LLMs to generate misaligned responses for harmful questions such as methods to create a bomb in school labs, recipes for harmful drugs, and ways to evade privacy rights. Another challenge is the multilingual capabilities of LLMs, which enable the model to understand and respond in multiple languages. Consequently, attackers exploit the unbalanced pre-training datasets of LLMs in different languages and the comparatively lower model performance in low-resource languages than high-resource ones. As a result, attackers use a low-resource languages to intentionally manipulate the model to create harmful responses. Many of the similar attack vectors have been patched by model providers, making the LLMs more robust against language-based manipulation. In this paper, we introduce a new black-box attack vector called the \emph{Sandwich attack}: a multi-language mixture attack, which manipulates state-of-the-art LLMs into generating harmful and misaligned responses. Our experiments with five different models, namely Google's Bard, Gemini Pro, LLaMA-2-70-B-Chat, GPT-3.5-Turbo, GPT-4, and Claude-3-OPUS, show that this attack vector can be used by adversaries to generate harmful responses and elicit misaligned responses from these models. By detailing both the mechanism and impact of the Sandwich attack, this paper aims to guide future research and development towards more secure and resilient LLMs, ensuring they serve the public good while minimizing potential for misuse.



## **43. $\textit{LinkPrompt}$: Natural and Universal Adversarial Attacks on Prompt-based Language Models**

cs.CL

Accepted to the main conference of NAACL2024

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2403.16432v3) [paper-pdf](http://arxiv.org/pdf/2403.16432v3)

**Authors**: Yue Xu, Wenjie Wang

**Abstract**: Prompt-based learning is a new language model training paradigm that adapts the Pre-trained Language Models (PLMs) to downstream tasks, which revitalizes the performance benchmarks across various natural language processing (NLP) tasks. Instead of using a fixed prompt template to fine-tune the model, some research demonstrates the effectiveness of searching for the prompt via optimization. Such prompt optimization process of prompt-based learning on PLMs also gives insight into generating adversarial prompts to mislead the model, raising concerns about the adversarial vulnerability of this paradigm. Recent studies have shown that universal adversarial triggers (UATs) can be generated to alter not only the predictions of the target PLMs but also the prediction of corresponding Prompt-based Fine-tuning Models (PFMs) under the prompt-based learning paradigm. However, UATs found in previous works are often unreadable tokens or characters and can be easily distinguished from natural texts with adaptive defenses. In this work, we consider the naturalness of the UATs and develop $\textit{LinkPrompt}$, an adversarial attack algorithm to generate UATs by a gradient-based beam search algorithm that not only effectively attacks the target PLMs and PFMs but also maintains the naturalness among the trigger tokens. Extensive results demonstrate the effectiveness of $\textit{LinkPrompt}$, as well as the transferability of UATs generated by $\textit{LinkPrompt}$ to open-sourced Large Language Model (LLM) Llama2 and API-accessed LLM GPT-3.5-turbo. The resource is available at $\href{https://github.com/SavannahXu79/LinkPrompt}{https://github.com/SavannahXu79/LinkPrompt}$.



## **44. LRR: Language-Driven Resamplable Continuous Representation against Adversarial Tracking Attacks**

cs.CV

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06247v1) [paper-pdf](http://arxiv.org/pdf/2404.06247v1)

**Authors**: Jianlang Chen, Xuhong Ren, Qing Guo, Felix Juefei-Xu, Di Lin, Wei Feng, Lei Ma, Jianjun Zhao

**Abstract**: Visual object tracking plays a critical role in visual-based autonomous systems, as it aims to estimate the position and size of the object of interest within a live video. Despite significant progress made in this field, state-of-the-art (SOTA) trackers often fail when faced with adversarial perturbations in the incoming frames. This can lead to significant robustness and security issues when these trackers are deployed in the real world. To achieve high accuracy on both clean and adversarial data, we propose building a spatial-temporal continuous representation using the semantic text guidance of the object of interest. This novel continuous representation enables us to reconstruct incoming frames to maintain semantic and appearance consistency with the object of interest and its clean counterparts. As a result, our proposed method successfully defends against different SOTA adversarial tracking attacks while maintaining high accuracy on clean data. In particular, our method significantly increases tracking accuracy under adversarial attacks with around 90% relative improvement on UAV123, which is even higher than the accuracy on clean data.



## **45. Towards Robust Domain Generation Algorithm Classification**

cs.CR

Accepted at ACM Asia Conference on Computer and Communications  Security (ASIA CCS 2024)

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06236v1) [paper-pdf](http://arxiv.org/pdf/2404.06236v1)

**Authors**: Arthur Drichel, Marc Meyer, Ulrike Meyer

**Abstract**: In this work, we conduct a comprehensive study on the robustness of domain generation algorithm (DGA) classifiers. We implement 32 white-box attacks, 19 of which are very effective and induce a false-negative rate (FNR) of $\approx$ 100\% on unhardened classifiers. To defend the classifiers, we evaluate different hardening approaches and propose a novel training scheme that leverages adversarial latent space vectors and discretized adversarial domains to significantly improve robustness. In our study, we highlight a pitfall to avoid when hardening classifiers and uncover training biases that can be easily exploited by attackers to bypass detection, but which can be mitigated by adversarial training (AT). In our study, we do not observe any trade-off between robustness and performance, on the contrary, hardening improves a classifier's detection performance for known and unknown DGAs. We implement all attacks and defenses discussed in this paper as a standalone library, which we make publicly available to facilitate hardening of DGA classifiers: https://gitlab.com/rwth-itsec/robust-dga-detection



## **46. FLEX: FLEXible Federated Learning Framework**

cs.CR

Submitted to Information Fusion

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06127v1) [paper-pdf](http://arxiv.org/pdf/2404.06127v1)

**Authors**: Francisco Herrera, Daniel Jiménez-López, Alberto Argente-Garrido, Nuria Rodríguez-Barroso, Cristina Zuheros, Ignacio Aguilera-Martos, Beatriz Bello, Mario García-Márquez, M. Victoria Luzón

**Abstract**: In the realm of Artificial Intelligence (AI), the need for privacy and security in data processing has become paramount. As AI applications continue to expand, the collection and handling of sensitive data raise concerns about individual privacy protection. Federated Learning (FL) emerges as a promising solution to address these challenges by enabling decentralized model training on local devices, thus preserving data privacy. This paper introduces FLEX: a FLEXible Federated Learning Framework designed to provide maximum flexibility in FL research experiments. By offering customizable features for data distribution, privacy parameters, and communication strategies, FLEX empowers researchers to innovate and develop novel FL techniques. The framework also includes libraries for specific FL implementations including: (1) anomalies, (2) blockchain, (3) adversarial attacks and defences, (4) natural language processing and (5) decision trees, enhancing its versatility and applicability in various domains. Overall, FLEX represents a significant advancement in FL research, facilitating the development of robust and efficient FL applications.



## **47. PeerAiD: Improving Adversarial Distillation from a Specialized Peer Tutor**

cs.LG

Accepted to CVPR 2024

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2403.06668v2) [paper-pdf](http://arxiv.org/pdf/2403.06668v2)

**Authors**: Jaewon Jung, Hongsun Jang, Jaeyong Song, Jinho Lee

**Abstract**: Adversarial robustness of the neural network is a significant concern when it is applied to security-critical domains. In this situation, adversarial distillation is a promising option which aims to distill the robustness of the teacher network to improve the robustness of a small student network. Previous works pretrain the teacher network to make it robust to the adversarial examples aimed at itself. However, the adversarial examples are dependent on the parameters of the target network. The fixed teacher network inevitably degrades its robustness against the unseen transferred adversarial examples which targets the parameters of the student network in the adversarial distillation process. We propose PeerAiD to make a peer network learn the adversarial examples of the student network instead of adversarial examples aimed at itself. PeerAiD is an adversarial distillation that trains the peer network and the student network simultaneously in order to make the peer network specialized for defending the student network. We observe that such peer networks surpass the robustness of pretrained robust teacher network against student-attacked adversarial samples. With this peer network and adversarial distillation, PeerAiD achieves significantly higher robustness of the student network with AutoAttack (AA) accuracy up to 1.66%p and improves the natural accuracy of the student network up to 4.72%p with ResNet-18 and TinyImageNet dataset.



## **48. Greedy-DiM: Greedy Algorithms for Unreasonably Effective Face Morphs**

cs.CV

Initial preprint. Under review

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06025v1) [paper-pdf](http://arxiv.org/pdf/2404.06025v1)

**Authors**: Zander W. Blasingame, Chen Liu

**Abstract**: Morphing attacks are an emerging threat to state-of-the-art Face Recognition (FR) systems, which aim to create a single image that contains the biometric information of multiple identities. Diffusion Morphs (DiM) are a recently proposed morphing attack that has achieved state-of-the-art performance for representation-based morphing attacks. However, none of the existing research on DiMs have leveraged the iterative nature of DiMs and left the DiM model as a black box, treating it no differently than one would a Generative Adversarial Network (GAN) or Varational AutoEncoder (VAE). We propose a greedy strategy on the iterative sampling process of DiM models which searches for an optimal step guided by an identity-based heuristic function. We compare our proposed algorithm against ten other state-of-the-art morphing algorithms using the open-source SYN-MAD 2022 competition dataset. We find that our proposed algorithm is unreasonably effective, fooling all of the tested FR systems with an MMPMR of 100%, outperforming all other morphing algorithms compared.



## **49. A Vulnerability of Attribution Methods Using Pre-Softmax Scores**

cs.LG

7 pages, 5 figures

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2307.03305v3) [paper-pdf](http://arxiv.org/pdf/2307.03305v3)

**Authors**: Miguel Lerma, Mirtha Lucas

**Abstract**: We discuss a vulnerability involving a category of attribution methods used to provide explanations for the outputs of convolutional neural networks working as classifiers. It is known that this type of networks are vulnerable to adversarial attacks, in which imperceptible perturbations of the input may alter the outputs of the model. In contrast, here we focus on effects that small modifications in the model may cause on the attribution method without altering the model outputs.



## **50. Improving the Accuracy-Robustness Trade-Off of Classifiers via Adaptive Smoothing**

cs.LG

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2301.12554v4) [paper-pdf](http://arxiv.org/pdf/2301.12554v4)

**Authors**: Yatong Bai, Brendon G. Anderson, Aerin Kim, Somayeh Sojoudi

**Abstract**: While prior research has proposed a plethora of methods that build neural classifiers robust against adversarial robustness, practitioners are still reluctant to adopt them due to their unacceptably severe clean accuracy penalties. This paper significantly alleviates this accuracy-robustness trade-off by mixing the output probabilities of a standard classifier and a robust classifier, where the standard network is optimized for clean accuracy and is not robust in general. We show that the robust base classifier's confidence difference for correct and incorrect examples is the key to this improvement. In addition to providing intuitions and empirical evidence, we theoretically certify the robustness of the mixed classifier under realistic assumptions. Furthermore, we adapt an adversarial input detector into a mixing network that adaptively adjusts the mixture of the two base models, further reducing the accuracy penalty of achieving robustness. The proposed flexible method, termed "adaptive smoothing", can work in conjunction with existing or even future methods that improve clean accuracy, robustness, or adversary detection. Our empirical evaluation considers strong attack methods, including AutoAttack and adaptive attack. On the CIFAR-100 dataset, our method achieves an 85.21% clean accuracy while maintaining a 38.72% $\ell_\infty$-AutoAttacked ($\epsilon = 8/255$) accuracy, becoming the second most robust method on the RobustBench CIFAR-100 benchmark as of submission, while improving the clean accuracy by ten percentage points compared with all listed models. The code that implements our method is available at https://github.com/Bai-YT/AdaptiveSmoothing.



