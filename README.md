# Latest Adversarial Attack Papers
**update at 2024-12-11 10:19:10**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Defending Against Neural Network Model Inversion Attacks via Data Poisoning**

cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07575v1) [paper-pdf](http://arxiv.org/pdf/2412.07575v1)

**Authors**: Shuai Zhou, Dayong Ye, Tianqing Zhu, Wanlei Zhou

**Abstract**: Model inversion attacks pose a significant privacy threat to machine learning models by reconstructing sensitive data from their outputs. While various defenses have been proposed to counteract these attacks, they often come at the cost of the classifier's utility, thus creating a challenging trade-off between privacy protection and model utility. Moreover, most existing defenses require retraining the classifier for enhanced robustness, which is impractical for large-scale, well-established models. This paper introduces a novel defense mechanism to better balance privacy and utility, particularly against adversaries who employ a machine learning model (i.e., inversion model) to reconstruct private data. Drawing inspiration from data poisoning attacks, which can compromise the performance of machine learning models, we propose a strategy that leverages data poisoning to contaminate the training data of inversion models, thereby preventing model inversion attacks.   Two defense methods are presented. The first, termed label-preserving poisoning attacks for all output vectors (LPA), involves subtle perturbations to all output vectors while preserving their labels. Our findings demonstrate that these minor perturbations, introduced through a data poisoning approach, significantly increase the difficulty of data reconstruction without compromising the utility of the classifier. Subsequently, we introduce a second method, label-flipping poisoning for partial output vectors (LFP), which selectively perturbs a small subset of output vectors and alters their labels during the process. Empirical results indicate that LPA is notably effective, outperforming the current state-of-the-art defenses. Our data poisoning-based defense provides a new retraining-free defense paradigm that preserves the victim classifier's utility.



## **2. Adaptive Epsilon Adversarial Training for Robust Gravitational Wave Parameter Estimation Using Normalizing Flows**

cs.LG

7 pages, 9 figures

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07559v1) [paper-pdf](http://arxiv.org/pdf/2412.07559v1)

**Authors**: Yiqian Yang, Xihua Zhu, Fan Zhang

**Abstract**: Adversarial training with Normalizing Flow (NF) models is an emerging research area aimed at improving model robustness through adversarial samples. In this study, we focus on applying adversarial training to NF models for gravitational wave parameter estimation. We propose an adaptive epsilon method for Fast Gradient Sign Method (FGSM) adversarial training, which dynamically adjusts perturbation strengths based on gradient magnitudes using logarithmic scaling. Our hybrid architecture, combining ResNet and Inverse Autoregressive Flow, reduces the Negative Log Likelihood (NLL) loss by 47\% under FGSM attacks compared to the baseline model, while maintaining an NLL of 4.2 on clean data (only 5\% higher than the baseline). For perturbation strengths between 0.01 and 0.1, our model achieves an average NLL of 5.8, outperforming both fixed-epsilon (NLL: 6.7) and progressive-epsilon (NLL: 7.2) methods. Under stronger Projected Gradient Descent attacks with perturbation strength of 0.05, our model maintains an NLL of 6.4, demonstrating superior robustness while avoiding catastrophic overfitting.



## **3. Quantifying the Prediction Uncertainty of Machine Learning Models for Individual Data**

cs.LG

PHD thesis

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07520v1) [paper-pdf](http://arxiv.org/pdf/2412.07520v1)

**Authors**: Koby Bibas

**Abstract**: Machine learning models have exhibited exceptional results in various domains. The most prevalent approach for learning is the empirical risk minimizer (ERM), which adapts the model's weights to reduce the loss on a training set and subsequently leverages these weights to predict the label for new test data. Nonetheless, ERM makes the assumption that the test distribution is similar to the training distribution, which may not always hold in real-world situations. In contrast, the predictive normalized maximum likelihood (pNML) was proposed as a min-max solution for the individual setting where no assumptions are made on the distribution of the tested input. This study investigates pNML's learnability for linear regression and neural networks, and demonstrates that pNML can improve the performance and robustness of these models on various tasks. Moreover, the pNML provides an accurate confidence measure for its output, showcasing state-of-the-art results for out-of-distribution detection, resistance to adversarial attacks, and active learning.



## **4. AHSG: Adversarial Attacks on High-level Semantics in Graph Neural Networks**

cs.LG

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07468v1) [paper-pdf](http://arxiv.org/pdf/2412.07468v1)

**Authors**: Kai Yuan, Xiaobing Pei, Haoran Yang

**Abstract**: Graph Neural Networks (GNNs) have garnered significant interest among researchers due to their impressive performance in graph learning tasks. However, like other deep neural networks, GNNs are also vulnerable to adversarial attacks. In existing adversarial attack methods for GNNs, the metric between the attacked graph and the original graph is usually the attack budget or a measure of global graph properties. However, we have found that it is possible to generate attack graphs that disrupt the primary semantics even within these constraints. To address this problem, we propose a Adversarial Attacks on High-level Semantics in Graph Neural Networks (AHSG), which is a graph structure attack model that ensures the retention of primary semantics. The latent representations of each node can extract rich semantic information by applying convolutional operations on graph data. These representations contain both task-relevant primary semantic information and task-irrelevant secondary semantic information. The latent representations of same-class nodes with the same primary semantics can fulfill the objective of modifying secondary semantics while preserving the primary semantics. Finally, the latent representations with attack effects is mapped to an attack graph using Projected Gradient Descent (PGD) algorithm. By attacking graph deep learning models with some advanced defense strategies, we validate that AHSG has superior attack effectiveness compared to other attack methods. Additionally, we employ Contextual Stochastic Block Models (CSBMs) as a proxy for the primary semantics to detect the attacked graph, confirming that AHSG almost does not disrupt the original primary semantics of the graph.



## **5. Addressing Key Challenges of Adversarial Attacks and Defenses in the Tabular Domain: A Methodological Framework for Coherence and Consistency**

cs.LG

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07326v1) [paper-pdf](http://arxiv.org/pdf/2412.07326v1)

**Authors**: Yael Itzhakev, Amit Giloni, Yuval Elovici, Asaf Shabtai

**Abstract**: Machine learning models trained on tabular data are vulnerable to adversarial attacks, even in realistic scenarios where attackers have access only to the model's outputs. Researchers evaluate such attacks by considering metrics like success rate, perturbation magnitude, and query count. However, unlike other data domains, the tabular domain contains complex interdependencies among features, presenting a unique aspect that should be evaluated: the need for the attack to generate coherent samples and ensure feature consistency for indistinguishability. Currently, there is no established methodology for evaluating adversarial samples based on these criteria. In this paper, we address this gap by proposing new evaluation criteria tailored for tabular attacks' quality; we defined anomaly-based framework to assess the distinguishability of adversarial samples and utilize the SHAP explainability technique to identify inconsistencies in the model's decision-making process caused by adversarial samples. These criteria could form the basis for potential detection methods and be integrated into established evaluation metrics for assessing attack's quality Additionally, we introduce a novel technique for perturbing dependent features while maintaining coherence and feature consistency within the sample. We compare different attacks' strategies, examining black-box query-based attacks and transferability-based gradient attacks across four target models. Our experiments, conducted on benchmark tabular datasets, reveal significant differences between the examined attacks' strategies in terms of the attacker's risk and effort and the attacks' quality. The findings provide valuable insights on the strengths, limitations, and trade-offs of various adversarial attacks in the tabular domain, laying a foundation for future research on attacks and defense development.



## **6. Backdoor Attacks against No-Reference Image Quality Assessment Models via A Scalable Trigger**

cs.CV

Accept by AAAI 2025

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07277v1) [paper-pdf](http://arxiv.org/pdf/2412.07277v1)

**Authors**: Yi Yu, Song Xia, Xun Lin, Wenhan Yang, Shijian Lu, Yap-peng Tan, Alex Kot

**Abstract**: No-Reference Image Quality Assessment (NR-IQA), responsible for assessing the quality of a single input image without using any reference, plays a critical role in evaluating and optimizing computer vision systems, e.g., low-light enhancement. Recent research indicates that NR-IQA models are susceptible to adversarial attacks, which can significantly alter predicted scores with visually imperceptible perturbations. Despite revealing vulnerabilities, these attack methods have limitations, including high computational demands, untargeted manipulation, limited practical utility in white-box scenarios, and reduced effectiveness in black-box scenarios. To address these challenges, we shift our focus to another significant threat and present a novel poisoning-based backdoor attack against NR-IQA (BAIQA), allowing the attacker to manipulate the IQA model's output to any desired target value by simply adjusting a scaling coefficient $\alpha$ for the trigger. We propose to inject the trigger in the discrete cosine transform (DCT) domain to improve the local invariance of the trigger for countering trigger diminishment in NR-IQA models due to widely adopted data augmentations. Furthermore, the universal adversarial perturbations (UAP) in the DCT space are designed as the trigger, to increase IQA model susceptibility to manipulation and improve attack effectiveness. In addition to the heuristic method for poison-label BAIQA (P-BAIQA), we explore the design of clean-label BAIQA (C-BAIQA), focusing on $\alpha$ sampling and image data refinement, driven by theoretical insights we reveal. Extensive experiments on diverse datasets and various NR-IQA models demonstrate the effectiveness of our attacks. Code will be released at https://github.com/yuyi-sd/BAIQA.



## **7. A Generative Victim Model for Segmentation**

cs.CV

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07274v1) [paper-pdf](http://arxiv.org/pdf/2412.07274v1)

**Authors**: Aixuan Li, Jing Zhang, Jiawei Shi, Yiran Zhong, Yuchao Dai

**Abstract**: We find that the well-trained victim models (VMs), against which the attacks are generated, serve as fundamental prerequisites for adversarial attacks, i.e. a segmentation VM is needed to generate attacks for segmentation. In this context, the victim model is assumed to be robust to achieve effective adversarial perturbation generation. Instead of focusing on improving the robustness of the task-specific victim models, we shift our attention to image generation. From an image generation perspective, we derive a novel VM for segmentation, aiming to generate adversarial perturbations for segmentation tasks without requiring models explicitly designed for image segmentation. Our approach to adversarial attack generation diverges from conventional white-box or black-box attacks, offering a fresh outlook on adversarial attack strategies. Experiments show that our attack method is able to generate effective adversarial attacks with good transferability.



## **8. CapGen:An Environment-Adaptive Generator of Adversarial Patches**

cs.CV

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07253v1) [paper-pdf](http://arxiv.org/pdf/2412.07253v1)

**Authors**: Chaoqun Li, Zhuodong Liu, Huanqian Yan, Hang Su

**Abstract**: Adversarial patches, often used to provide physical stealth protection for critical assets and assess perception algorithm robustness, usually neglect the need for visual harmony with the background environment, making them easily noticeable. Moreover, existing methods primarily concentrate on improving attack performance, disregarding the intricate dynamics of adversarial patch elements. In this work, we introduce the Camouflaged Adversarial Pattern Generator (CAPGen), a novel approach that leverages specific base colors from the surrounding environment to produce patches that seamlessly blend with their background for superior visual stealthiness while maintaining robust adversarial performance. We delve into the influence of both patterns (i.e., color-agnostic texture information) and colors on the effectiveness of attacks facilitated by patches, discovering that patterns exert a more pronounced effect on performance than colors. Based on these findings, we propose a rapid generation strategy for adversarial patches. This involves updating the colors of high-performance adversarial patches to align with those of the new environment, ensuring visual stealthiness without compromising adversarial impact. This paper is the first to comprehensively examine the roles played by patterns and colors in the context of adversarial patches.



## **9. Adversarial Filtering Based Evasion and Backdoor Attacks to EEG-Based Brain-Computer Interfaces**

cs.HC

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07231v1) [paper-pdf](http://arxiv.org/pdf/2412.07231v1)

**Authors**: Lubin Meng, Xue Jiang, Xiaoqing Chen, Wenzhong Liu, Hanbin Luo, Dongrui Wu

**Abstract**: A brain-computer interface (BCI) enables direct communication between the brain and an external device. Electroencephalogram (EEG) is a common input signal for BCIs, due to its convenience and low cost. Most research on EEG-based BCIs focuses on the accurate decoding of EEG signals, while ignoring their security. Recent studies have shown that machine learning models in BCIs are vulnerable to adversarial attacks. This paper proposes adversarial filtering based evasion and backdoor attacks to EEG-based BCIs, which are very easy to implement. Experiments on three datasets from different BCI paradigms demonstrated the effectiveness of our proposed attack approaches. To our knowledge, this is the first study on adversarial filtering for EEG-based BCIs, raising a new security concern and calling for more attention on the security of BCIs.



## **10. A Parametric Approach to Adversarial Augmentation for Cross-Domain Iris Presentation Attack Detection**

cs.CV

IEEE/CVF Winter Conference on Applications of Computer Vision (WACV),  2025

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07199v1) [paper-pdf](http://arxiv.org/pdf/2412.07199v1)

**Authors**: Debasmita Pal, Redwan Sony, Arun Ross

**Abstract**: Iris-based biometric systems are vulnerable to presentation attacks (PAs), where adversaries present physical artifacts (e.g., printed iris images, textured contact lenses) to defeat the system. This has led to the development of various presentation attack detection (PAD) algorithms, which typically perform well in intra-domain settings. However, they often struggle to generalize effectively in cross-domain scenarios, where training and testing employ different sensors, PA instruments, and datasets. In this work, we use adversarial training samples of both bonafide irides and PAs to improve the cross-domain performance of a PAD classifier. The novelty of our approach lies in leveraging transformation parameters from classical data augmentation schemes (e.g., translation, rotation) to generate adversarial samples. We achieve this through a convolutional autoencoder, ADV-GEN, that inputs original training samples along with a set of geometric and photometric transformations. The transformation parameters act as regularization variables, guiding ADV-GEN to generate adversarial samples in a constrained search space. Experiments conducted on the LivDet-Iris 2017 database, comprising four datasets, and the LivDet-Iris 2020 dataset, demonstrate the efficacy of our proposed method. The code is available at https://github.com/iPRoBe-lab/ADV-GEN-IrisPAD.



## **11. PrisonBreak: Jailbreaking Large Language Models with Fewer Than Twenty-Five Targeted Bit-flips**

cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07192v1) [paper-pdf](http://arxiv.org/pdf/2412.07192v1)

**Authors**: Zachary Coalson, Jeonghyun Woo, Shiyang Chen, Yu Sun, Lishan Yang, Prashant Nair, Bo Fang, Sanghyun Hong

**Abstract**: We introduce a new class of attacks on commercial-scale (human-aligned) language models that induce jailbreaking through targeted bitwise corruptions in model parameters. Our adversary can jailbreak billion-parameter language models with fewer than 25 bit-flips in all cases$-$and as few as 5 in some$-$using up to 40$\times$ less bit-flips than existing attacks on computer vision models at least 100$\times$ smaller. Unlike prompt-based jailbreaks, our attack renders these models in memory 'uncensored' at runtime, allowing them to generate harmful responses without any input modifications. Our attack algorithm efficiently identifies target bits to flip, offering up to 20$\times$ more computational efficiency than previous methods. This makes it practical for language models with billions of parameters. We show an end-to-end exploitation of our attack using software-induced fault injection, Rowhammer (RH). Our work examines 56 DRAM RH profiles from DDR4 and LPDDR4X devices with different RH vulnerabilities. We show that our attack can reliably induce jailbreaking in systems similar to those affected by prior bit-flip attacks. Moreover, our approach remains effective even against highly RH-secure systems (e.g., 46$\times$ more secure than previously tested systems). Our analyses further reveal that: (1) models with less post-training alignment require fewer bit flips to jailbreak; (2) certain model components, such as value projection layers, are substantially more vulnerable than others; and (3) our method is mechanistically different than existing jailbreaks. Our findings highlight a pressing, practical threat to the language model ecosystem and underscore the need for research to protect these models from bit-flip attacks.



## **12. dSTAR: Straggler Tolerant and Byzantine Resilient Distributed SGD**

cs.DC

15 pages

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07151v1) [paper-pdf](http://arxiv.org/pdf/2412.07151v1)

**Authors**: Jiahe Yan, Pratik Chaudhari, Leonard Kleinrock

**Abstract**: Distributed model training needs to be adapted to challenges such as the straggler effect and Byzantine attacks. When coordinating the training process with multiple computing nodes, ensuring timely and reliable gradient aggregation amidst network and system malfunctions is essential. To tackle these issues, we propose \textit{dSTAR}, a lightweight and efficient approach for distributed stochastic gradient descent (SGD) that enhances robustness and convergence. \textit{dSTAR} selectively aggregates gradients by collecting updates from the first \(k\) workers to respond, filtering them based on deviations calculated using an ensemble median. This method not only mitigates the impact of stragglers but also fortifies the model against Byzantine adversaries. We theoretically establish that \textit{dSTAR} is (\(\alpha, f\))-Byzantine resilient and achieves a linear convergence rate. Empirical evaluations across various scenarios demonstrate that \textit{dSTAR} consistently maintains high accuracy, outperforming other Byzantine-resilient methods that often suffer up to a 40-50\% accuracy drop under attack. Our results highlight \textit{dSTAR} as a robust solution for training models in distributed environments prone to both straggler delays and Byzantine faults.



## **13. Defensive Dual Masking for Robust Adversarial Defense**

cs.CL

First version

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07078v1) [paper-pdf](http://arxiv.org/pdf/2412.07078v1)

**Authors**: Wangli Yang, Jie Yang, Yi Guo, Johan Barthelemy

**Abstract**: The field of textual adversarial defenses has gained considerable attention in recent years due to the increasing vulnerability of natural language processing (NLP) models to adversarial attacks, which exploit subtle perturbations in input text to deceive models. This paper introduces the Defensive Dual Masking (DDM) algorithm, a novel approach designed to enhance model robustness against such attacks. DDM utilizes a unique adversarial training strategy where [MASK] tokens are strategically inserted into training samples to prepare the model to handle adversarial perturbations more effectively. During inference, potentially adversarial tokens are dynamically replaced with [MASK] tokens to neutralize potential threats while preserving the core semantics of the input. The theoretical foundation of our approach is explored, demonstrating how the selective masking mechanism strengthens the model's ability to identify and mitigate adversarial manipulations. Our empirical evaluation across a diverse set of benchmark datasets and attack mechanisms consistently shows that DDM outperforms state-of-the-art defense techniques, improving model accuracy and robustness. Moreover, when applied to Large Language Models (LLMs), DDM also enhances their resilience to adversarial attacks, providing a scalable defense mechanism for large-scale NLP applications.



## **14. Dense Cross-Connected Ensemble Convolutional Neural Networks for Enhanced Model Robustness**

cs.CV

6 pages, 1 figure

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.07022v1) [paper-pdf](http://arxiv.org/pdf/2412.07022v1)

**Authors**: Longwei Wang, Xueqian Li, Zheng Zhang

**Abstract**: The resilience of convolutional neural networks against input variations and adversarial attacks remains a significant challenge in image recognition tasks. Motivated by the need for more robust and reliable image recognition systems, we propose the Dense Cross-Connected Ensemble Convolutional Neural Network (DCC-ECNN). This novel architecture integrates the dense connectivity principle of DenseNet with the ensemble learning strategy, incorporating intermediate cross-connections between different DenseNet paths to facilitate extensive feature sharing and integration. The DCC-ECNN architecture leverages DenseNet's efficient parameter usage and depth while benefiting from the robustness of ensemble learning, ensuring a richer and more resilient feature representation.



## **15. Fiat-Shamir for Proofs Lacks a Proof Even in the Presence of Shared Entanglement**

quant-ph

58 pages, 4 figures; accepted in Quantum

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2204.02265v5) [paper-pdf](http://arxiv.org/pdf/2204.02265v5)

**Authors**: Frédéric Dupuis, Philippe Lamontagne, Louis Salvail

**Abstract**: We explore the cryptographic power of arbitrary shared physical resources. The most general such resource is access to a fresh entangled quantum state at the outset of each protocol execution. We call this the Common Reference Quantum State (CRQS) model, in analogy to the well-known Common Reference String (CRS). The CRQS model is a natural generalization of the CRS model but appears to be more powerful: in the two-party setting, a CRQS can sometimes exhibit properties associated with a Random Oracle queried once by measuring a maximally entangled state in one of many mutually unbiased bases. We formalize this notion as a Weak One-Time Random Oracle (WOTRO), where we only ask of the $m$-bit output to have some randomness when conditioned on the $n$-bit input.   We show that when $n-m\in\omega(\lg n)$, any protocol for WOTRO in the CRQS model can be attacked by an (inefficient) adversary. Moreover, our adversary is efficiently simulatable, which rules out the possibility of proving the computational security of a scheme by a fully black-box reduction to a cryptographic game assumption. On the other hand, we introduce a non-game quantum assumption for hash functions that implies WOTRO in the CRQS model (where the CRQS consists only of EPR pairs). We first build a statistically secure WOTRO protocol where $m=n$, then hash the output.   The impossibility of WOTRO has the following consequences. First, we show the fully-black-box impossibility of a quantum Fiat-Shamir transform, extending the impossibility result of Bitansky et al. (TCC 2013) to the CRQS model. Second, we show a fully-black-box impossibility result for a strenghtened version of quantum lightning (Zhandry, Eurocrypt 2019) where quantum bolts have an additional parameter that cannot be changed without generating new bolts. Our results also apply to $2$-message protocols in the plain model.



## **16. WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs**

cs.CL

NeurIPS 2024 Camera Ready. First two authors contributed equally.  Third and fourth authors contributed equally

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2406.18495v3) [paper-pdf](http://arxiv.org/pdf/2406.18495v3)

**Authors**: Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, Nouha Dziri

**Abstract**: We introduce WildGuard -- an open, light-weight moderation tool for LLM safety that achieves three goals: (1) identifying malicious intent in user prompts, (2) detecting safety risks of model responses, and (3) determining model refusal rate. Together, WildGuard serves the increasing needs for automatic safety moderation and evaluation of LLM interactions, providing a one-stop tool with enhanced accuracy and broad coverage across 13 risk categories. While existing open moderation tools such as Llama-Guard2 score reasonably well in classifying straightforward model interactions, they lag far behind a prompted GPT-4, especially in identifying adversarial jailbreaks and in evaluating models' refusals, a key measure for evaluating safety behaviors in model responses.   To address these challenges, we construct WildGuardMix, a large-scale and carefully balanced multi-task safety moderation dataset with 92K labeled examples that cover vanilla (direct) prompts and adversarial jailbreaks, paired with various refusal and compliance responses. WildGuardMix is a combination of WildGuardTrain, the training data of WildGuard, and WildGuardTest, a high-quality human-annotated moderation test set with 5K labeled items covering broad risk scenarios. Through extensive evaluations on WildGuardTest and ten existing public benchmarks, we show that WildGuard establishes state-of-the-art performance in open-source safety moderation across all the three tasks compared to ten strong existing open-source moderation models (e.g., up to 26.4% improvement on refusal detection). Importantly, WildGuard matches and sometimes exceeds GPT-4 performance (e.g., up to 3.9% improvement on prompt harmfulness identification). WildGuard serves as a highly effective safety moderator in an LLM interface, reducing the success rate of jailbreak attacks from 79.8% to 2.4%.



## **17. Take Fake as Real: Realistic-like Robust Black-box Adversarial Attack to Evade AIGC Detection**

cs.CV

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06727v1) [paper-pdf](http://arxiv.org/pdf/2412.06727v1)

**Authors**: Caiyun Xie, Dengpan Ye, Yunming Zhang, Long Tang, Yunna Lv, Jiacheng Deng, Jiawei Song

**Abstract**: The security of AI-generated content (AIGC) detection based on GANs and diffusion models is closely related to the credibility of multimedia content. Malicious adversarial attacks can evade these developing AIGC detection. However, most existing adversarial attacks focus only on GAN-generated facial images detection, struggle to be effective on multi-class natural images and diffusion-based detectors, and exhibit poor invisibility. To fill this gap, we first conduct an in-depth analysis of the vulnerability of AIGC detectors and discover the feature that detectors vary in vulnerability to different post-processing. Then, considering the uncertainty of detectors in real-world scenarios, and based on the discovery, we propose a Realistic-like Robust Black-box Adversarial attack (R$^2$BA) with post-processing fusion optimization. Unlike typical perturbations, R$^2$BA uses real-world post-processing, i.e., Gaussian blur, JPEG compression, Gaussian noise and light spot to generate adversarial examples. Specifically, we use a stochastic particle swarm algorithm with inertia decay to optimize post-processing fusion intensity and explore the detector's decision boundary. Guided by the detector's fake probability, R$^2$BA enhances/weakens the detector-vulnerable/detector-robust post-processing intensity to strike a balance between adversariality and invisibility. Extensive experiments on popular/commercial AIGC detectors and datasets demonstrate that R$^2$BA exhibits impressive anti-detection performance, excellent invisibility, and strong robustness in GAN-based and diffusion-based cases. Compared to state-of-the-art white-box and black-box attacks, R$^2$BA shows significant improvements of 15% and 21% in anti-detection performance under the original and robust scenario respectively, offering valuable insights for the security of AIGC detection in real-world applications.



## **18. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

cs.CR

15 pages, 13 figures

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2202.03195v6) [paper-pdf](http://arxiv.org/pdf/2202.03195v6)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for a different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against one defense. We find that both attacks are robust against the investigated defense, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.



## **19. Vulnerability, Where Art Thou? An Investigation of Vulnerability Management in Android Smartphone Chipsets**

cs.CR

Accepted by Network and Distributed System Security (NDSS) Symposium  2025

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06556v1) [paper-pdf](http://arxiv.org/pdf/2412.06556v1)

**Authors**: Daniel Klischies, Philipp Mackensen, Veelasha Moonsamy

**Abstract**: Vulnerabilities in Android smartphone chipsets have severe consequences, as recent real-world attacks have demonstrated that adversaries can leverage vulnerabilities to execute arbitrary code or exfiltrate confidential information. Despite the far-reaching impact of such attacks, the lifecycle of chipset vulnerabilities has yet to be investigated, with existing papers primarily investigating vulnerabilities in the Android operating system. This paper provides a comprehensive and empirical study of the current state of smartphone chipset vulnerability management within the Android ecosystem. For the first time, we create a unified knowledge base of 3,676 chipset vulnerabilities affecting 437 chipset models from all four major chipset manufacturers, combined with 6,866 smartphone models. Our analysis revealed that the same vulnerabilities are often included in multiple generations of chipsets, providing novel empirical evidence that vulnerabilities are inherited through multiple chipset generations. Furthermore, we demonstrate that the commonly accepted 90-day responsible vulnerability disclosure period is seldom adhered to. We find that a single vulnerability often affects hundreds to thousands of different smartphone models, for which update availability is, as we show, often unclear or heavily delayed. Leveraging the new insights gained from our empirical analysis, we recommend several changes that chipset manufacturers can implement to improve the security posture of their products. At the same time, our knowledge base enables academic researchers to conduct more representative evaluations of smartphone chipsets, accurately assess the impact of vulnerabilities they discover, and identify avenues for future research.



## **20. Flexible and Scalable Deep Dendritic Spiking Neural Networks with Multiple Nonlinear Branching**

cs.NE

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06355v1) [paper-pdf](http://arxiv.org/pdf/2412.06355v1)

**Authors**: Yifan Huang, Wei Fang, Zhengyu Ma, Guoqi Li, Yonghong Tian

**Abstract**: Recent advances in spiking neural networks (SNNs) have a predominant focus on network architectures, while relatively little attention has been paid to the underlying neuron model. The point neuron models, a cornerstone of deep SNNs, pose a bottleneck on the network-level expressivity since they depict somatic dynamics only. In contrast, the multi-compartment models in neuroscience offer remarkable expressivity by introducing dendritic morphology and dynamics, but remain underexplored in deep learning due to their unaffordable computational cost and inflexibility. To combine the advantages of both sides for a flexible, efficient yet more powerful model, we propose the dendritic spiking neuron (DendSN) incorporating multiple dendritic branches with nonlinear dynamics. Compared to the point spiking neurons, DendSN exhibits significantly higher expressivity. DendSN's flexibility enables its seamless integration into diverse deep SNN architectures. To accelerate dendritic SNNs (DendSNNs), we parallelize dendritic state updates across time steps, and develop Triton kernels for GPU-level acceleration. As a result, we can construct large-scale DendSNNs with depth comparable to their point SNN counterparts. Next, we comprehensively evaluate DendSNNs' performance on various demanding tasks. By modulating dendritic branch strengths using a context signal, catastrophic forgetting of DendSNNs is substantially mitigated. Moreover, DendSNNs demonstrate enhanced robustness against noise and adversarial attacks compared to point SNNs, and excel in few-shot learning settings. Our work firstly demonstrates the possibility of training bio-plausible dendritic SNNs with depths and scales comparable to traditional point SNNs, and reveals superior expressivity and robustness of reduced dendritic neuron models in deep learning, thereby offering a fresh perspective on advancing neural network design.



## **21. SmartReco: Detecting Read-Only Reentrancy via Fine-Grained Cross-DApp Analysis**

cs.SE

Accepted by ICSE 2025

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2409.18468v2) [paper-pdf](http://arxiv.org/pdf/2409.18468v2)

**Authors**: Jingwen Zhang, Zibin Zheng, Yuhong Nan, Mingxi Ye, Kaiwen Ning, Yu Zhang, Weizhe Zhang

**Abstract**: Despite the increasing popularity of Decentralized Applications (DApps), they are suffering from various vulnerabilities that can be exploited by adversaries for profits. Among such vulnerabilities, Read-Only Reentrancy (called ROR in this paper), is an emerging type of vulnerability that arises from the complex interactions between DApps. In the recent three years, attack incidents of ROR have already caused around 30M USD losses to the DApp ecosystem. Existing techniques for vulnerability detection in smart contracts can hardly detect Read-Only Reentrancy attacks, due to the lack of tracking and analyzing the complex interactions between multiple DApps. In this paper, we propose SmartReco, a new framework for detecting Read-Only Reentrancy vulnerability in DApps through a novel combination of static and dynamic analysis (i.e., fuzzing) over smart contracts. The key design behind SmartReco is threefold: (1) SmartReco identifies the boundary between different DApps from the heavy-coupled cross-contract interactions. (2) SmartReco performs fine-grained static analysis to locate points of interest (i.e., entry functions) that may lead to ROR. (3) SmartReco utilizes the on-chain transaction data and performs multi-function fuzzing (i.e., the entry function and victim function) across different DApps to verify the existence of ROR. Our evaluation of a manual-labeled dataset with 45 RORs shows that SmartReco achieves a precision of 88.63% and a recall of 86.36%. In addition, SmartReco successfully detects 43 new RORs from 123 popular DApps. The total assets affected by such RORs reach around 520,000 USD.



## **22. Unidirectional focusing of light using structured diffractive surfaces**

physics.optics

20 Pages, 6 Figures

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06221v1) [paper-pdf](http://arxiv.org/pdf/2412.06221v1)

**Authors**: Yuhang Li, Tianyi Gan, Jingxi Li, Mona Jarrahi, Aydogan Ozcan

**Abstract**: Unidirectional optical systems enable selective control of light through asymmetric processing of radiation, effectively transmitting light in one direction while blocking unwanted propagation in the opposite direction. Here, we introduce a reciprocal diffractive unidirectional focusing design based on linear and isotropic diffractive layers that are structured. Using deep learning-based optimization, a cascaded set of diffractive layers are spatially engineered at the wavelength scale to focus light efficiently in the forward direction while blocking it in the opposite direction. The forward energy focusing efficiency and the backward energy suppression capabilities of this unidirectional architecture were demonstrated under various illumination angles and wavelengths, illustrating the versatility of our polarization-insensitive design. Furthermore, we demonstrated that these designs are resilient to adversarial attacks that utilize wavefront engineering from outside. Experimental validation using terahertz radiation confirmed the feasibility of this diffractive unidirectional focusing framework. Diffractive unidirectional designs can operate across different parts of the electromagnetic spectrum by scaling the resulting diffractive features proportional to the wavelength of light and will find applications in security, defense, and optical communication, among others.



## **23. A Real-Time Defense Against Object Vanishing Adversarial Patch Attacks for Object Detection in Autonomous Vehicles**

cs.CV

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06215v1) [paper-pdf](http://arxiv.org/pdf/2412.06215v1)

**Authors**: Jaden Mu

**Abstract**: Autonomous vehicles (AVs) increasingly use DNN-based object detection models in vision-based perception. Correct detection and classification of obstacles is critical to ensure safe, trustworthy driving decisions. Adversarial patches aim to fool a DNN with intentionally generated patterns concentrated in a localized region of an image. In particular, object vanishing patch attacks can cause object detection models to fail to detect most or all objects in a scene, posing a significant practical threat to AVs.   This work proposes ADAV (Adversarial Defense for Autonomous Vehicles), a novel defense methodology against object vanishing patch attacks specifically designed for autonomous vehicles. Unlike existing defense methods which have high latency or are designed for static images, ADAV runs in real-time and leverages contextual information from prior frames in an AV's video feed. ADAV checks if the object detector's output for the target frame is temporally consistent with the output from a previous reference frame to detect the presence of a patch. If the presence of a patch is detected, ADAV uses gradient-based attribution to localize adversarial pixels that break temporal consistency. This two stage procedure allows ADAV to efficiently process clean inputs, and both stages are optimized to be low latency. ADAV is evaluated using real-world driving data from the Berkeley Deep Drive BDD100K dataset, and demonstrates high adversarial and clean performance.



## **24. Credible fusion of evidence in distributed system subject to cyberattacks**

cs.CR

29 pages, 11 figures

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.04496v2) [paper-pdf](http://arxiv.org/pdf/2412.04496v2)

**Authors**: Chaoxiong Ma, Yan Liang

**Abstract**: Given that distributed systems face adversarial behaviors such as eavesdropping and cyberattacks, how to ensure the evidence fusion result is credible becomes a must-be-addressed topic. Different from traditional research that assumes nodes are cooperative, we focus on three requirements for evidence fusion, i.e., preserving evidence's privacy, identifying attackers and excluding their evidence, and dissipating high-conflicting among evidence caused by random noise and interference. To this end, this paper proposes an algorithm for credible evidence fusion against cyberattacks. Firstly, the fusion strategy is constructed based on conditionalized credibility to avoid counterintuitive fusion results caused by high-conflicting. Under this strategy, distributed evidence fusion is transformed into the average consensus problem for the weighted average value by conditional credibility of multi-source evidence (WAVCCME), which implies a more concise consensus process and lower computational complexity than existing algorithms. Secondly, a state decomposition and reconstruction strategy with weight encryption is designed, and its effectiveness for privacy-preserving under directed graphs is guaranteed: decomposing states into different random sub-states for different neighbors to defend against internal eavesdroppers, and encrypting the sub-states' weight in the reconstruction to guard against out-of-system eavesdroppers. Finally, the identities and types of attackers are identified by inter-neighbor broadcasting and comparison of nodes' states, and the proposed update rule with state corrections is used to achieve the consensus of the WAVCCME. The states of normal nodes are shown to converge to their WAVCCME, while the attacker's evidence is excluded from the fusion, as verified by the simulation on a distributed unmanned reconnaissance swarm.



## **25. Privacy-Preserving Large Language Models: Mechanisms, Applications, and Future Directions**

cs.CR

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06113v1) [paper-pdf](http://arxiv.org/pdf/2412.06113v1)

**Authors**: Guoshenghui Zhao, Eric Song

**Abstract**: The rapid advancement of large language models (LLMs) has revolutionized natural language processing, enabling applications in diverse domains such as healthcare, finance and education. However, the growing reliance on extensive data for training and inference has raised significant privacy concerns, ranging from data leakage to adversarial attacks. This survey comprehensively explores the landscape of privacy-preserving mechanisms tailored for LLMs, including differential privacy, federated learning, cryptographic protocols, and trusted execution environments. We examine their efficacy in addressing key privacy challenges, such as membership inference and model inversion attacks, while balancing trade-offs between privacy and model utility. Furthermore, we analyze privacy-preserving applications of LLMs in privacy-sensitive domains, highlighting successful implementations and inherent limitations. Finally, this survey identifies emerging research directions, emphasizing the need for novel frameworks that integrate privacy by design into the lifecycle of LLMs. By synthesizing state-of-the-art approaches and future trends, this paper provides a foundation for developing robust, privacy-preserving large language models that safeguard sensitive information without compromising performance.



## **26. TrojanForge: Generating Adversarial Hardware Trojan Examples Using Reinforcement Learning**

cs.CR

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2405.15184v3) [paper-pdf](http://arxiv.org/pdf/2405.15184v3)

**Authors**: Amin Sarihi, Peter Jamieson, Ahmad Patooghy, Abdel-Hameed A. Badawy

**Abstract**: The Hardware Trojan (HT) problem can be thought of as a continuous game between attackers and defenders, each striving to outsmart the other by leveraging any available means for an advantage. Machine Learning (ML) has recently played a key role in advancing HT research. Various novel techniques, such as Reinforcement Learning (RL) and Graph Neural Networks (GNNs), have shown HT insertion and detection capabilities. HT insertion with ML techniques, specifically, has seen a spike in research activity due to the shortcomings of conventional HT benchmarks and the inherent human design bias that occurs when we create them. This work continues this innovation by presenting a tool called TrojanForge, capable of generating HT adversarial examples that defeat HT detectors; demonstrating the capabilities of GAN-like adversarial tools for automatic HT insertion. We introduce an RL environment where the RL insertion agent interacts with HT detectors in an insertion-detection loop where the agent collects rewards based on its success in bypassing HT detectors. Our results show that this process helps inserted HTs evade various HT detectors, achieving high attack success percentages. This tool provides insight into why HT insertion fails in some instances and how we can leverage this knowledge in defense.



## **27. Anti-Reference: Universal and Immediate Defense Against Reference-Based Generation**

cs.CV

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2412.05980v1) [paper-pdf](http://arxiv.org/pdf/2412.05980v1)

**Authors**: Yiren Song, Shengtao Lou, Xiaokang Liu, Hai Ci, Pei Yang, Jiaming Liu, Mike Zheng Shou

**Abstract**: Diffusion models have revolutionized generative modeling with their exceptional ability to produce high-fidelity images. However, misuse of such potent tools can lead to the creation of fake news or disturbing content targeting individuals, resulting in significant social harm. In this paper, we introduce Anti-Reference, a novel method that protects images from the threats posed by reference-based generation techniques by adding imperceptible adversarial noise to the images. We propose a unified loss function that enables joint attacks on fine-tuning-based customization methods, non-fine-tuning customization methods, and human-centric driving methods. Based on this loss, we train a Adversarial Noise Encoder to predict the noise or directly optimize the noise using the PGD method. Our method shows certain transfer attack capabilities, effectively challenging both gray-box models and some commercial APIs. Extensive experiments validate the performance of Anti-Reference, establishing a new benchmark in image security.



## **28. Revisiting DeepFool: generalization and improvement**

cs.LG

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2303.12481v2) [paper-pdf](http://arxiv.org/pdf/2303.12481v2)

**Authors**: Alireza Abdollahpoorrostam, Mahed Abroshan, Seyed-Mohsen Moosavi-Dezfooli

**Abstract**: Deep neural networks have been known to be vulnerable to adversarial examples, which are inputs that are modified slightly to fool the network into making incorrect predictions. This has led to a significant amount of research on evaluating the robustness of these networks against such perturbations. One particularly important robustness metric is the robustness to minimal $\ell_2$ adversarial perturbations. However, existing methods for evaluating this robustness metric are either computationally expensive or not very accurate. In this paper, we introduce a new family of adversarial attacks that strike a balance between effectiveness and computational efficiency. Our proposed attacks are generalizations of the well-known DeepFool (DF) attack, while they remain simple to understand and implement. We demonstrate that our attacks outperform existing methods in terms of both effectiveness and computational efficiency. Our proposed attacks are also suitable for evaluating the robustness of large models and can be used to perform adversarial training (AT) to achieve state-of-the-art robustness to minimal $\ell_2$ adversarial perturbations.



## **29. TrojanRobot: Backdoor Attacks Against LLM-based Embodied Robots in the Physical World**

cs.RO

Initial version with preliminary results. We welcome any feedback or  suggestions

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2411.11683v2) [paper-pdf](http://arxiv.org/pdf/2411.11683v2)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Yichen Wang, Wei Wan, Aishan Liu, Leo Yu Zhang

**Abstract**: Robotic manipulation refers to the autonomous handling and interaction of robots with objects using advanced techniques in robotics and artificial intelligence. The advent of powerful tools such as large language models (LLMs) and large vision-language models (LVLMs) has significantly enhanced the capabilities of these robots in environmental perception and decision-making. However, the introduction of these intelligent agents has led to security threats such as jailbreak attacks and adversarial attacks.   In this research, we take a further step by proposing a backdoor attack specifically targeting robotic manipulation and, for the first time, implementing backdoor attack in the physical world. By embedding a backdoor visual language model into the visual perception module within the robotic system, we successfully mislead the robotic arm's operation in the physical world, given the presence of common items as triggers. Experimental evaluations in the physical world demonstrate the effectiveness of the proposed backdoor attack.



## **30. Adversarial Transferability in Deep Denoising Models: Theoretical Insights and Robustness Enhancement via Out-of-Distribution Typical Set Sampling**

cs.CV

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2412.05943v1) [paper-pdf](http://arxiv.org/pdf/2412.05943v1)

**Authors**: Jie Ning, Jiebao Sun, Shengzhu Shi, Zhichang Guo, Yao Li, Hongwei Li, Boying Wu

**Abstract**: Deep learning-based image denoising models demonstrate remarkable performance, but their lack of robustness analysis remains a significant concern. A major issue is that these models are susceptible to adversarial attacks, where small, carefully crafted perturbations to input data can cause them to fail. Surprisingly, perturbations specifically crafted for one model can easily transfer across various models, including CNNs, Transformers, unfolding models, and plug-and-play models, leading to failures in those models as well. Such high adversarial transferability is not observed in classification models. We analyze the possible underlying reasons behind the high adversarial transferability through a series of hypotheses and validation experiments. By characterizing the manifolds of Gaussian noise and adversarial perturbations using the concept of typical set and the asymptotic equipartition property, we prove that adversarial samples deviate slightly from the typical set of the original input distribution, causing the models to fail. Based on these insights, we propose a novel adversarial defense method: the Out-of-Distribution Typical Set Sampling Training strategy (TS). TS not only significantly enhances the model's robustness but also marginally improves denoising performance compared to the original model.



## **31. BAMBA: A Bimodal Adversarial Multi-Round Black-Box Jailbreak Attacker for LVLMs**

cs.CR

A Bimodal Adversarial Multi-Round Black-Box Jailbreak Attacker for  LVLMs

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2412.05892v1) [paper-pdf](http://arxiv.org/pdf/2412.05892v1)

**Authors**: Ruoxi Cheng, Yizhong Ding, Shuirong Cao, Shaowei Yuan, Zhiqiang Wang, Xiaojun Jia

**Abstract**: LVLMs are widely used but vulnerable to illegal or unethical responses under jailbreak attacks. To ensure their responsible deployment in real-world applications, it is essential to understand their vulnerabilities. There are four main issues in current work: single-round attack limitation, insufficient dual-modal synergy, poor transferability to black-box models, and reliance on prompt engineering. To address these limitations, we propose BAMBA, a bimodal adversarial multi-round black-box jailbreak attacker for LVLMs. We first use an image optimizer to learn malicious features from a harmful corpus, then deepen these features through a bimodal optimizer through text-image interaction, generating adversarial text and image for jailbreak. Experiments on various LVLMs and datasets demonstrate that BAMBA outperforms other baselines.



## **32. Understanding the Impact of Graph Reduction on Adversarial Robustness in Graph Neural Networks**

cs.LG

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2412.05883v1) [paper-pdf](http://arxiv.org/pdf/2412.05883v1)

**Authors**: Kerui Wu, Ka-Ho Chow, Wenqi Wei, Lei Yu

**Abstract**: As Graph Neural Networks (GNNs) become increasingly popular for learning from large-scale graph data across various domains, their susceptibility to adversarial attacks when using graph reduction techniques for scalability remains underexplored. In this paper, we present an extensive empirical study to investigate the impact of graph reduction techniques, specifically graph coarsening and sparsification, on the robustness of GNNs against adversarial attacks. Through extensive experiments involving multiple datasets and GNN architectures, we examine the effects of four sparsification and six coarsening methods on the poisoning attacks. Our results indicate that, while graph sparsification can mitigate the effectiveness of certain poisoning attacks, such as Mettack, it has limited impact on others, like PGD. Conversely, graph coarsening tends to amplify the adversarial impact, significantly reducing classification accuracy as the reduction ratio decreases. Additionally, we provide a novel analysis of the causes driving these effects and examine how defensive GNN models perform under graph reduction, offering practical insights for designing robust GNNs within graph acceleration systems.



## **33. DeMem: Privacy-Enhanced Robust Adversarial Learning via De-Memorization**

cs.LG

10 pages

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.05767v2) [paper-pdf](http://arxiv.org/pdf/2412.05767v2)

**Authors**: Xiaoyu Luo, Qiongxiu Li

**Abstract**: Adversarial robustness, the ability of a model to withstand manipulated inputs that cause errors, is essential for ensuring the trustworthiness of machine learning models in real-world applications. However, previous studies have shown that enhancing adversarial robustness through adversarial training increases vulnerability to privacy attacks. While differential privacy can mitigate these attacks, it often compromises robustness against both natural and adversarial samples. Our analysis reveals that differential privacy disproportionately impacts low-risk samples, causing an unintended performance drop. To address this, we propose DeMem, which selectively targets high-risk samples, achieving a better balance between privacy protection and model robustness. DeMem is versatile and can be seamlessly integrated into various adversarial training techniques. Extensive evaluations across multiple training methods and datasets demonstrate that DeMem significantly reduces privacy leakage while maintaining robustness against both natural and adversarial samples. These results confirm DeMem's effectiveness and broad applicability in enhancing privacy without compromising robustness.



## **34. Query-Based Adversarial Prompt Generation**

cs.CL

**SubmitDate**: 2024-12-07    [abs](http://arxiv.org/abs/2402.12329v2) [paper-pdf](http://arxiv.org/pdf/2402.12329v2)

**Authors**: Jonathan Hayase, Ema Borevkovic, Nicholas Carlini, Florian Tramèr, Milad Nasr

**Abstract**: Recent work has shown it is possible to construct adversarial examples that cause an aligned language model to emit harmful strings or perform harmful behavior. Existing attacks work either in the white-box setting (with full access to the model weights), or through transferability: the phenomenon that adversarial examples crafted on one model often remain effective on other models. We improve on prior work with a query-based attack that leverages API access to a remote language model to construct adversarial examples that cause the model to emit harmful strings with (much) higher probability than with transfer-only attacks. We validate our attack on GPT-3.5 and OpenAI's safety classifier; we can cause GPT-3.5 to emit harmful strings that current transfer attacks fail at, and we can evade the safety classifier with nearly 100% probability.



## **35. REGE: A Method for Incorporating Uncertainty in Graph Embeddings**

cs.LG

**SubmitDate**: 2024-12-07    [abs](http://arxiv.org/abs/2412.05735v1) [paper-pdf](http://arxiv.org/pdf/2412.05735v1)

**Authors**: Zohair Shafi, Germans Savcisens, Tina Eliassi-Rad

**Abstract**: Machine learning models for graphs in real-world applications are prone to two primary types of uncertainty: (1) those that arise from incomplete and noisy data and (2) those that arise from uncertainty of the model in its output. These sources of uncertainty are not mutually exclusive. Additionally, models are susceptible to targeted adversarial attacks, which exacerbate both of these uncertainties. In this work, we introduce Radius Enhanced Graph Embeddings (REGE), an approach that measures and incorporates uncertainty in data to produce graph embeddings with radius values that represent the uncertainty of the model's output. REGE employs curriculum learning to incorporate data uncertainty and conformal learning to address the uncertainty in the model's output. In our experiments, we show that REGE's graph embeddings perform better under adversarial attacks by an average of 1.5% (accuracy) against state-of-the-art methods.



## **36. PrivAgent: Agentic-based Red-teaming for LLM Privacy Leakage**

cs.CR

**SubmitDate**: 2024-12-07    [abs](http://arxiv.org/abs/2412.05734v1) [paper-pdf](http://arxiv.org/pdf/2412.05734v1)

**Authors**: Yuzhou Nie, Zhun Wang, Ye Yu, Xian Wu, Xuandong Zhao, Wenbo Guo, Dawn Song

**Abstract**: Recent studies have discovered that LLMs have serious privacy leakage concerns, where an LLM may be fooled into outputting private information under carefully crafted adversarial prompts. These risks include leaking system prompts, personally identifiable information, training data, and model parameters. Most existing red-teaming approaches for privacy leakage rely on humans to craft the adversarial prompts. A few automated methods are proposed for system prompt extraction, but they cannot be applied to more severe risks (e.g., training data extraction) and have limited effectiveness even for system prompt extraction.   In this paper, we propose PrivAgent, a novel black-box red-teaming framework for LLM privacy leakage. We formulate different risks as a search problem with a unified attack goal. Our framework trains an open-source LLM through reinforcement learning as the attack agent to generate adversarial prompts for different target models under different risks. We propose a novel reward function to provide effective and fine-grained rewards for the attack agent. Finally, we introduce customizations to better fit our general framework to system prompt extraction and training data extraction. Through extensive evaluations, we first show that PrivAgent outperforms existing automated methods in system prompt leakage against six popular LLMs. Notably, our approach achieves a 100% success rate in extracting system prompts from real-world applications in OpenAI's GPT Store. We also show PrivAgent's effectiveness in extracting training data from an open-source LLM with a success rate of 5.9%. We further demonstrate PrivAgent's effectiveness in evading the existing guardrail defense and its helpfulness in enabling better safety alignment. Finally, we validate our customized designs through a detailed ablation study. We release our code here https://github.com/rucnyz/RedAgent.



## **37. Nearly Solved? Robust Deepfake Detection Requires More than Visual Forensics**

cs.CV

**SubmitDate**: 2024-12-07    [abs](http://arxiv.org/abs/2412.05676v1) [paper-pdf](http://arxiv.org/pdf/2412.05676v1)

**Authors**: Guy Levy, Nathan Liebmann

**Abstract**: Deepfakes are on the rise, with increased sophistication and prevalence allowing for high-profile social engineering attacks. Detecting them in the wild is therefore important as ever, giving rise to new approaches breaking benchmark records in this task. In line with previous work, we show that recently developed state-of-the-art detectors are susceptible to classical adversarial attacks, even in a highly-realistic black-box setting, putting their usability in question. We argue that crucial 'robust features' of deepfakes are in their higher semantics, and follow that with evidence that a detector based on a semantic embedding model is less susceptible to black-box perturbation attacks. We show that large visuo-lingual models like GPT-4o can perform zero-shot deepfake detection better than current state-of-the-art methods, and introduce a novel attack based on high-level semantic manipulation. Finally, we argue that hybridising low- and high-level detectors can improve adversarial robustness, based on their complementary strengths and weaknesses.



## **38. From Flexibility to Manipulation: The Slippery Slope of XAI Evaluation**

cs.AI

Published in ECCV 2024 Workshop on Explainable Computer Vision: Where  are We and Where are We Going? Shorter non-archival version also appeared in  the NeurIPS 2024 Interpretable AI workshop. Code is available at  \url{https://github.com/Wickstrom/quantitative-xai-manipulation}

**SubmitDate**: 2024-12-07    [abs](http://arxiv.org/abs/2412.05592v1) [paper-pdf](http://arxiv.org/pdf/2412.05592v1)

**Authors**: Kristoffer Wickstrøm, Marina Marie-Claire Höhne, Anna Hedström

**Abstract**: The lack of ground truth explanation labels is a fundamental challenge for quantitative evaluation in explainable artificial intelligence (XAI). This challenge becomes especially problematic when evaluation methods have numerous hyperparameters that must be specified by the user, as there is no ground truth to determine an optimal hyperparameter selection. It is typically not feasible to do an exhaustive search of hyperparameters so researchers typically make a normative choice based on similar studies in the literature, which provides great flexibility for the user. In this work, we illustrate how this flexibility can be exploited to manipulate the evaluation outcome. We frame this manipulation as an adversarial attack on the evaluation where seemingly innocent changes in hyperparameter setting significantly influence the evaluation outcome. We demonstrate the effectiveness of our manipulation across several datasets with large changes in evaluation outcomes across several explanation methods and models. Lastly, we propose a mitigation strategy based on ranking across hyperparameters that aims to provide robustness towards such manipulation. This work highlights the difficulty of conducting reliable XAI evaluation and emphasizes the importance of a holistic and transparent approach to evaluation in XAI.



## **39. Practical Region-level Attack against Segment Anything Models**

cs.CV

Code is released at https://github.com/ShenYifanS/S-RA_T-RA

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2404.08255v2) [paper-pdf](http://arxiv.org/pdf/2404.08255v2)

**Authors**: Yifan Shen, Zhengyuan Li, Gang Wang

**Abstract**: Segment Anything Models (SAM) have made significant advancements in image segmentation, allowing users to segment target portions of an image with a single click (i.e., user prompt). Given its broad applications, the robustness of SAM against adversarial attacks is a critical concern. While recent works have explored adversarial attacks against a pre-defined prompt/click, their threat model is not yet realistic: (1) they often assume the user-click position is known to the attacker (point-based attack), and (2) they often operate under a white-box setting with limited transferability. In this paper, we propose a more practical region-level attack where attackers do not need to know the precise user prompt. The attack remains effective as the user clicks on any point on the target object in the image, hiding the object from SAM. Also, by adapting a spectrum transformation method, we make the attack more transferable under a black-box setting. Both control experiments and testing against real-world SAM services confirm its effectiveness.



## **40. LIAR: Leveraging Alignment (Best-of-N) to Jailbreak LLMs in Seconds**

cs.CL

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2412.05232v1) [paper-pdf](http://arxiv.org/pdf/2412.05232v1)

**Authors**: James Beetham, Souradip Chakraborty, Mengdi Wang, Furong Huang, Amrit Singh Bedi, Mubarak Shah

**Abstract**: Many existing jailbreak techniques rely on solving discrete combinatorial optimization, while more recent approaches involve training LLMs to generate multiple adversarial prompts. However, both approaches require significant computational resources to produce even a single adversarial prompt. We hypothesize that the inefficiency of current approaches stems from an inadequate characterization of the jailbreak problem. To address this gap, we formulate the jailbreak problem in terms of alignment. By starting from an available safety-aligned model, we leverage an unsafe reward to guide the safe model towards generating unsafe outputs using alignment techniques (e.g., reinforcement learning from human feedback), effectively performing jailbreaking via alignment. We propose a novel jailbreak method called LIAR (LeveragIng Alignment to jailbReak). To demonstrate the simplicity and effectiveness of our approach, we employ a best-of-N method to solve the alignment problem. LIAR offers significant advantages: lower computational requirements without additional training, fully black-box operation, competitive attack success rates, and more human-readable prompts. We provide theoretical insights into the possibility of jailbreaking a safety-aligned model, revealing inherent vulnerabilities in current alignment strategies for LLMs. We also provide sub-optimality guarantees for the proposed \algo. Experimentally, we achieve ASR comparable to the SoTA with a 10x improvement to perplexity and a Time-to-Attack measured in seconds rather than tens of hours.



## **41. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

cs.CL

8 pages. Submitted to ARR October cycle

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2412.05139v1) [paper-pdf](http://arxiv.org/pdf/2412.05139v1)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, GPTID, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0\%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.



## **42. On Borrowed Time -- Preventing Static Side-Channel Analysis**

cs.CR

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2307.09001v2) [paper-pdf](http://arxiv.org/pdf/2307.09001v2)

**Authors**: Robert Dumitru, Thorben Moos, Andrew Wabnitz, Yuval Yarom

**Abstract**: In recent years a new class of side-channel attacks has emerged. Instead of targeting device emissions during dynamic computation, adversaries now frequently exploit the leakage or response behaviour of integrated circuits in a static state. Members of this class include Static Power Side-Channel Analysis (SCA), Laser Logic State Imaging (LLSI) and Impedance Analysis (IA). Despite relying on different physical phenomena, they all enable the extraction of sensitive information from circuits in a static state with high accuracy and low noise -- a trait that poses a significant threat to many established side-channel countermeasures.   In this work, we point out the shortcomings of existing solutions and derive a simple yet effective countermeasure. We observe that in order to realise their full potential, static side-channel attacks require the targeted data to remain unchanged for a certain amount of time. For some cryptographic secrets this happens naturally, for others it requires stopping the target circuit's clock. Our proposal, called Borrowed Time, hinders an attacker's ability to leverage such idle conditions, even if full control over the global clock signal is obtained. For that, by design, key-dependent data may only be present in unprotected temporary storage when strictly needed. Borrowed Time then continuously monitors the target circuit and upon detecting an idle state, securely wipes sensitive contents.   We demonstrate the need for our countermeasure and its effectiveness by mounting practical static power SCA attacks against cryptographic systems on FPGAs, with and without Borrowed Time. In one case we attack a masked implementation and show that it is only protected with our countermeasure in place. Furthermore we demonstrate that secure on-demand wiping of sensitive data works as intended, affirming the theory that the technique also effectively hinders LLSI and IA.



## **43. MultiTrust: A Comprehensive Benchmark Towards Trustworthy Multimodal Large Language Models**

cs.CL

100 pages, 84 figures, 33 tables

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2406.07057v2) [paper-pdf](http://arxiv.org/pdf/2406.07057v2)

**Authors**: Yichi Zhang, Yao Huang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Yifan Wang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, Jun Zhu

**Abstract**: Despite the superior capabilities of Multimodal Large Language Models (MLLMs) across diverse tasks, they still face significant trustworthiness challenges. Yet, current literature on the assessment of trustworthy MLLMs remains limited, lacking a holistic evaluation to offer thorough insights into future improvements. In this work, we establish MultiTrust, the first comprehensive and unified benchmark on the trustworthiness of MLLMs across five primary aspects: truthfulness, safety, robustness, fairness, and privacy. Our benchmark employs a rigorous evaluation strategy that addresses both multimodal risks and cross-modal impacts, encompassing 32 diverse tasks with self-curated datasets. Extensive experiments with 21 modern MLLMs reveal some previously unexplored trustworthiness issues and risks, highlighting the complexities introduced by the multimodality and underscoring the necessity for advanced methodologies to enhance their reliability. For instance, typical proprietary models still struggle with the perception of visually confusing images and are vulnerable to multimodal jailbreaking and adversarial attacks; MLLMs are more inclined to disclose privacy in text and reveal ideological and cultural biases even when paired with irrelevant images in inference, indicating that the multimodality amplifies the internal risks from base LLMs. Additionally, we release a scalable toolbox for standardized trustworthiness research, aiming to facilitate future advancements in this important field. Code and resources are publicly available at: https://multi-trust.github.io/.



## **44. Quantum Security Analysis of the Key-Alternating Ciphers**

quant-ph

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2412.05026v1) [paper-pdf](http://arxiv.org/pdf/2412.05026v1)

**Authors**: Chen Bai, Mehdi Esmaili, Atul Mantri

**Abstract**: We study the security of key-alternating ciphers (KAC), a generalization of Even-Mansour ciphers over multiple rounds, which serve as abstractions for many block cipher constructions, particularly AES. While the classical security of KAC has been extensively studied, little is known about its security against quantum adversaries. In this paper, we introduce the first nontrivial quantum key-recovery attack on multi-round KAC in a model where the adversary has quantum access to only one of the public permutations. Our attack applies to any $t$-round KAC, achieving quantum query complexity of $O(2^{\frac{t(t+1)n}{(t+1)^2+1}})$, where $n$ is the size of each individual key, in a realistic quantum threat model, compared to the classical bound of $O(2^{\frac{tn}{(t+1)}})$ queries given by Bogdanev et al. (EUROCRYPT 2012). Our quantum attack leverages a novel approach based on quantum walk algorithms. Additionally, using the quantum hybrid method in our new threat model, we extend the Even-Mansour lower bound of $\Omega(2^{\frac{n}{3}})$ given by Alagic et al. (EUROCRYPT 2022) to $\Omega(2^{\frac{(t-1)n}{t}})$ for the $t$-round KAC (for $t \geq 2$).



## **45. Endless Jailbreaks with Bijection Learning**

cs.CL

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2410.01294v2) [paper-pdf](http://arxiv.org/pdf/2410.01294v2)

**Authors**: Brian R. Y. Huang, Maximilian Li, Leonard Tang

**Abstract**: Despite extensive safety measures, LLMs are vulnerable to adversarial inputs, or jailbreaks, which can elicit unsafe behaviors. In this work, we introduce bijection learning, a powerful attack algorithm which automatically fuzzes LLMs for safety vulnerabilities using randomly-generated encodings whose complexity can be tightly controlled. We leverage in-context learning to teach models bijective encodings, pass encoded queries to the model to bypass built-in safety mechanisms, and finally decode responses back into English. Our attack is extremely effective on a wide range of frontier language models. Moreover, by controlling complexity parameters such as number of key-value mappings in the encodings, we find a close relationship between the capability level of the attacked LLM and the average complexity of the most effective bijection attacks. Our work highlights that new vulnerabilities in frontier models can emerge with scale: more capable models are more severely jailbroken by bijection attacks.



## **46. SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-image Diffusion Models**

cs.CV

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2412.04852v1) [paper-pdf](http://arxiv.org/pdf/2412.04852v1)

**Authors**: Zilan Wang, Junfeng Guo, Jiacheng Zhu, Yiming Li, Heng Huang, Muhao Chen, Zhengzhong Tu

**Abstract**: Recent advances in large-scale text-to-image (T2I) diffusion models have enabled a variety of downstream applications, including style customization, subject-driven personalization, and conditional generation. As T2I models require extensive data and computational resources for training, they constitute highly valued intellectual property (IP) for their legitimate owners, yet making them incentive targets for unauthorized fine-tuning by adversaries seeking to leverage these models for customized, usually profitable applications. Existing IP protection methods for diffusion models generally involve embedding watermark patterns and then verifying ownership through generated outputs examination, or inspecting the model's feature space. However, these techniques are inherently ineffective in practical scenarios when the watermarked model undergoes fine-tuning, and the feature space is inaccessible during verification ((i.e., black-box setting). The model is prone to forgetting the previously learned watermark knowledge when it adapts to a new task. To address this challenge, we propose SleeperMark, a novel framework designed to embed resilient watermarks into T2I diffusion models. SleeperMark explicitly guides the model to disentangle the watermark information from the semantic concepts it learns, allowing the model to retain the embedded watermark while continuing to be fine-tuned to new downstream tasks. Our extensive experiments demonstrate the effectiveness of SleeperMark across various types of diffusion models, including latent diffusion models (e.g., Stable Diffusion) and pixel diffusion models (e.g., DeepFloyd-IF), showing robustness against downstream fine-tuning and various attacks at both the image and model levels, with minimal impact on the model's generative capability. The code is available at https://github.com/taco-group/SleeperMark.



## **47. Plentiful Jailbreaks with String Compositions**

cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2411.01084v2) [paper-pdf](http://arxiv.org/pdf/2411.01084v2)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or red-teamers, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary string compositions, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.



## **48. PADetBench: Towards Benchmarking Physical Attacks against Object Detection**

cs.CV

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2408.09181v2) [paper-pdf](http://arxiv.org/pdf/2408.09181v2)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Lap-Pui Chau, Shaohui Mei

**Abstract**: Physical attacks against object detection have gained increasing attention due to their significant practical implications. However, conducting physical experiments is extremely time-consuming and labor-intensive. Moreover, physical dynamics and cross-domain transformation are challenging to strictly regulate in the real world, leading to unaligned evaluation and comparison, severely hindering the development of physically robust models. To accommodate these challenges, we explore utilizing realistic simulation to thoroughly and rigorously benchmark physical attacks with fairness under controlled physical dynamics and cross-domain transformation. This resolves the problem of capturing identical adversarial images that cannot be achieved in the real world. Our benchmark includes 20 physical attack methods, 48 object detectors, comprehensive physical dynamics, and evaluation metrics. We also provide end-to-end pipelines for dataset generation, detection, evaluation, and further analysis. In addition, we perform 8064 groups of evaluation based on our benchmark, which includes both overall evaluation and further detailed ablation studies for controlled physical dynamics. Through these experiments, we provide in-depth analyses of physical attack performance and physical adversarial robustness, draw valuable observations, and discuss potential directions for future research.   Codebase: https://github.com/JiaweiLian/Benchmarking_Physical_Attack



## **49. Defending Object Detectors against Patch Attacks with Out-of-Distribution Smoothing**

cs.LG

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2205.08989v2) [paper-pdf](http://arxiv.org/pdf/2205.08989v2)

**Authors**: Ryan Feng, Neal Mangaokar, Jihye Choi, Somesh Jha, Atul Prakash

**Abstract**: Patch attacks against object detectors have been of recent interest due to their being physically realizable and more closely aligned with practical systems. In response to this threat, many new defenses have been proposed that train a patch segmenter model to detect and remove the patch before the image is passed to the downstream model. We unify these approaches with a flexible framework, OODSmoother, which characterizes the properties of approaches that aim to remove adversarial patches. This framework naturally guides us to design 1) a novel adaptive attack that breaks existing patch attack defenses on object detectors, and 2) a novel defense approach SemPrior that takes advantage of semantic priors. Our key insight behind SemPrior is that the existing machine learning-based patch detectors struggle to learn semantic priors and that explicitly incorporating them can improve performance. We find that SemPrior alone provides up to a 40% gain, or up to a 60% gain when combined with existing defenses.



## **50. Targeting the Core: A Simple and Effective Method to Attack RAG-based Agents via Direct LLM Manipulation**

cs.AI

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.04415v1) [paper-pdf](http://arxiv.org/pdf/2412.04415v1)

**Authors**: Xuying Li, Zhuo Li, Yuji Kosuga, Yasuhiro Yoshida, Victor Bian

**Abstract**: AI agents, powered by large language models (LLMs), have transformed human-computer interactions by enabling seamless, natural, and context-aware communication. While these advancements offer immense utility, they also inherit and amplify inherent safety risks such as bias, fairness, hallucinations, privacy breaches, and a lack of transparency. This paper investigates a critical vulnerability: adversarial attacks targeting the LLM core within AI agents. Specifically, we test the hypothesis that a deceptively simple adversarial prefix, such as \textit{Ignore the document}, can compel LLMs to produce dangerous or unintended outputs by bypassing their contextual safeguards. Through experimentation, we demonstrate a high attack success rate (ASR), revealing the fragility of existing LLM defenses. These findings emphasize the urgent need for robust, multi-layered security measures tailored to mitigate vulnerabilities at the LLM level and within broader agent-based architectures.



