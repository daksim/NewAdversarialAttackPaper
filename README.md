# Latest Adversarial Attack Papers
**update at 2023-10-30 09:53:46**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. LipSim: A Provably Robust Perceptual Similarity Metric**

cs.CV

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2310.18274v1) [paper-pdf](http://arxiv.org/pdf/2310.18274v1)

**Authors**: Sara Ghazanfari, Alexandre Araujo, Prashanth Krishnamurthy, Farshad Khorrami, Siddharth Garg

**Abstract**: Recent years have seen growing interest in developing and applying perceptual similarity metrics. Research has shown the superiority of perceptual metrics over pixel-wise metrics in aligning with human perception and serving as a proxy for the human visual system. On the other hand, as perceptual metrics rely on neural networks, there is a growing concern regarding their resilience, given the established vulnerability of neural networks to adversarial attacks. It is indeed logical to infer that perceptual metrics may inherit both the strengths and shortcomings of neural networks. In this work, we demonstrate the vulnerability of state-of-the-art perceptual similarity metrics based on an ensemble of ViT-based feature extractors to adversarial attacks. We then propose a framework to train a robust perceptual similarity metric called LipSim (Lipschitz Similarity Metric) with provable guarantees. By leveraging 1-Lipschitz neural networks as the backbone, LipSim provides guarded areas around each data point and certificates for all perturbations within an $\ell_2$ ball. Finally, a comprehensive set of experiments shows the performance of LipSim in terms of natural and certified scores and on the image retrieval application. The code is available at https://github.com/SaraGhazanfari/LipSim.



## **2. $α$-Mutual Information: A Tunable Privacy Measure for Privacy Protection in Data Sharing**

cs.LG

2023 22nd IEEE International Conference on Machine Learning and  Applications (ICMLA)

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2310.18241v1) [paper-pdf](http://arxiv.org/pdf/2310.18241v1)

**Authors**: MirHamed Jafarzadeh Asl, Mohammadhadi Shateri, Fabrice Labeau

**Abstract**: This paper adopts Arimoto's $\alpha$-Mutual Information as a tunable privacy measure, in a privacy-preserving data release setting that aims to prevent disclosing private data to adversaries. By fine-tuning the privacy metric, we demonstrate that our approach yields superior models that effectively thwart attackers across various performance dimensions. We formulate a general distortion-based mechanism that manipulates the original data to offer privacy protection. The distortion metrics are determined according to the data structure of a specific experiment. We confront the problem expressed in the formulation by employing a general adversarial deep learning framework that consists of a releaser and an adversary, trained with opposite goals. This study conducts empirical experiments on images and time-series data to verify the functionality of $\alpha$-Mutual Information. We evaluate the privacy-utility trade-off of customized models and compare them to mutual information as the baseline measure. Finally, we analyze the consequence of an attacker's access to side information about private data and witness that adapting the privacy measure results in a more refined model than the state-of-the-art in terms of resiliency against side information.



## **3. Adaptive Webpage Fingerprinting from TLS Traces**

cs.CR

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2010.10294v2) [paper-pdf](http://arxiv.org/pdf/2010.10294v2)

**Authors**: Vasilios Mavroudis, Jamie Hayes

**Abstract**: In webpage fingerprinting, an on-path adversary infers the specific webpage loaded by a victim user by analysing the patterns in the encrypted TLS traffic exchanged between the user's browser and the website's servers. This work studies modern webpage fingerprinting adversaries against the TLS protocol; aiming to shed light on their capabilities and inform potential defences. Despite the importance of this research area (the majority of global Internet users rely on standard web browsing with TLS) and the potential real-life impact, most past works have focused on attacks specific to anonymity networks (e.g., Tor). We introduce a TLS-specific model that: 1) scales to an unprecedented number of target webpages, 2) can accurately classify thousands of classes it never encountered during training, and 3) has low operational costs even in scenarios of frequent page updates. Based on these findings, we then discuss TLS-specific countermeasures and evaluate the effectiveness of the existing padding capabilities provided by TLS 1.3.



## **4. Elevating Code-mixed Text Handling through Auditory Information of Words**

cs.CL

Accepted to EMNLP 2023

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2310.18155v1) [paper-pdf](http://arxiv.org/pdf/2310.18155v1)

**Authors**: Mamta, Zishan Ahmad, Asif Ekbal

**Abstract**: With the growing popularity of code-mixed data, there is an increasing need for better handling of this type of data, which poses a number of challenges, such as dealing with spelling variations, multiple languages, different scripts, and a lack of resources. Current language models face difficulty in effectively handling code-mixed data as they primarily focus on the semantic representation of words and ignore the auditory phonetic features. This leads to difficulties in handling spelling variations in code-mixed text. In this paper, we propose an effective approach for creating language models for handling code-mixed textual data using auditory information of words from SOUNDEX. Our approach includes a pre-training step based on masked-language-modelling, which includes SOUNDEX representations (SAMLM) and a new method of providing input data to the pre-trained model. Through experimentation on various code-mixed datasets (of different languages) for sentiment, offensive and aggression classification tasks, we establish that our novel language modeling approach (SAMLM) results in improved robustness towards adversarial attacks on code-mixed classification tasks. Additionally, our SAMLM based approach also results in better classification results over the popular baselines for code-mixed tasks. We use the explainability technique, SHAP (SHapley Additive exPlanations) to explain how the auditory features incorporated through SAMLM assist the model to handle the code-mixed text effectively and increase robustness against adversarial attacks \footnote{Source code has been made available on \url{https://github.com/20118/DefenseWithPhonetics}, \url{https://www.iitp.ac.in/~ai-nlp-ml/resources.html\#Phonetics}}.



## **5. A Unified Algebraic Perspective on Lipschitz Neural Networks**

cs.LG

ICLR 2023. Spotlight paper

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2303.03169v2) [paper-pdf](http://arxiv.org/pdf/2303.03169v2)

**Authors**: Alexandre Araujo, Aaron Havens, Blaise Delattre, Alexandre Allauzen, Bin Hu

**Abstract**: Important research efforts have focused on the design and training of neural networks with a controlled Lipschitz constant. The goal is to increase and sometimes guarantee the robustness against adversarial attacks. Recent promising techniques draw inspirations from different backgrounds to design 1-Lipschitz neural networks, just to name a few: convex potential layers derive from the discretization of continuous dynamical systems, Almost-Orthogonal-Layer proposes a tailored method for matrix rescaling. However, it is today important to consider the recent and promising contributions in the field under a common theoretical lens to better design new and improved layers. This paper introduces a novel algebraic perspective unifying various types of 1-Lipschitz neural networks, including the ones previously mentioned, along with methods based on orthogonality and spectral methods. Interestingly, we show that many existing techniques can be derived and generalized via finding analytical solutions of a common semidefinite programming (SDP) condition. We also prove that AOL biases the scaled weight to the ones which are close to the set of orthogonal matrices in a certain mathematical manner. Moreover, our algebraic condition, combined with the Gershgorin circle theorem, readily leads to new and diverse parameterizations for 1-Lipschitz network layers. Our approach, called SDP-based Lipschitz Layers (SLL), allows us to design non-trivial yet efficient generalization of convex potential layers. Finally, the comprehensive set of experiments on image classification shows that SLLs outperform previous approaches on certified robust accuracy. Code is available at https://github.com/araujoalexandre/Lipschitz-SLL-Networks.



## **6. Defending Against Transfer Attacks From Public Models**

cs.LG

Under submission. Code available at  https://github.com/wagner-group/pubdef

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17645v1) [paper-pdf](http://arxiv.org/pdf/2310.17645v1)

**Authors**: Chawin Sitawarin, Jaewon Chang, David Huang, Wesson Altoyan, David Wagner

**Abstract**: Adversarial attacks have been a looming and unaddressed threat in the industry. However, through a decade-long history of the robustness evaluation literature, we have learned that mounting a strong or optimal attack is challenging. It requires both machine learning and domain expertise. In other words, the white-box threat model, religiously assumed by a large majority of the past literature, is unrealistic. In this paper, we propose a new practical threat model where the adversary relies on transfer attacks through publicly available surrogate models. We argue that this setting will become the most prevalent for security-sensitive applications in the future. We evaluate the transfer attacks in this setting and propose a specialized defense method based on a game-theoretic perspective. The defenses are evaluated under 24 public models and 11 attack algorithms across three datasets (CIFAR-10, CIFAR-100, and ImageNet). Under this threat model, our defense, PubDef, outperforms the state-of-the-art white-box adversarial training by a large margin with almost no loss in the normal accuracy. For instance, on ImageNet, our defense achieves 62% accuracy under the strongest transfer attack vs only 36% of the best adversarially trained model. Its accuracy when not under attack is only 2% lower than that of an undefended model (78% vs 80%). We release our code at https://github.com/wagner-group/pubdef.



## **7. A Survey on Transferability of Adversarial Examples across Deep Neural Networks**

cs.CV

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17626v1) [paper-pdf](http://arxiv.org/pdf/2310.17626v1)

**Authors**: Jindong Gu, Xiaojun Jia, Pau de Jorge, Wenqain Yu, Xinwei Liu, Avery Ma, Yuan Xun, Anjun Hu, Ashkan Khakzar, Zhijiang Li, Xiaochun Cao, Philip Torr

**Abstract**: The emergence of Deep Neural Networks (DNNs) has revolutionized various domains, enabling the resolution of complex tasks spanning image recognition, natural language processing, and scientific problem-solving. However, this progress has also exposed a concerning vulnerability: adversarial examples. These crafted inputs, imperceptible to humans, can manipulate machine learning models into making erroneous predictions, raising concerns for safety-critical applications. An intriguing property of this phenomenon is the transferability of adversarial examples, where perturbations crafted for one model can deceive another, often with a different architecture. This intriguing property enables "black-box" attacks, circumventing the need for detailed knowledge of the target model. This survey explores the landscape of the adversarial transferability of adversarial examples. We categorize existing methodologies to enhance adversarial transferability and discuss the fundamental principles guiding each approach. While the predominant body of research primarily concentrates on image classification, we also extend our discussion to encompass other vision tasks and beyond. Challenges and future prospects are discussed, highlighting the importance of fortifying DNNs against adversarial vulnerabilities in an evolving landscape.



## **8. Gaussian Membership Inference Privacy**

cs.LG

NeurIPS 2023 camera-ready. The first two authors contributed equally

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2306.07273v2) [paper-pdf](http://arxiv.org/pdf/2306.07273v2)

**Authors**: Tobias Leemann, Martin Pawelczyk, Gjergji Kasneci

**Abstract**: We propose a novel and practical privacy notion called $f$-Membership Inference Privacy ($f$-MIP), which explicitly considers the capabilities of realistic adversaries under the membership inference attack threat model. Consequently, $f$-MIP offers interpretable privacy guarantees and improved utility (e.g., better classification accuracy). In particular, we derive a parametric family of $f$-MIP guarantees that we refer to as $\mu$-Gaussian Membership Inference Privacy ($\mu$-GMIP) by theoretically analyzing likelihood ratio-based membership inference attacks on stochastic gradient descent (SGD). Our analysis highlights that models trained with standard SGD already offer an elementary level of MIP. Additionally, we show how $f$-MIP can be amplified by adding noise to gradient updates. Our analysis further yields an analytical membership inference attack that offers two distinct advantages over previous approaches. First, unlike existing state-of-the-art attacks that require training hundreds of shadow models, our attack does not require any shadow model. Second, our analytical attack enables straightforward auditing of our privacy notion $f$-MIP. Finally, we quantify how various hyperparameters (e.g., batch size, number of model parameters) and specific data characteristics determine an attacker's ability to accurately infer a point's membership in the training set. We demonstrate the effectiveness of our method on models trained on vision and tabular datasets.



## **9. Instability of computer vision models is a necessary result of the task itself**

cs.CV

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17559v1) [paper-pdf](http://arxiv.org/pdf/2310.17559v1)

**Authors**: Oliver Turnbull, George Cevora

**Abstract**: Adversarial examples resulting from instability of current computer vision models are an extremely important topic due to their potential to compromise any application. In this paper we demonstrate that instability is inevitable due to a) symmetries (translational invariance) of the data, b) the categorical nature of the classification task, and c) the fundamental discrepancy of classifying images as objects themselves. The issue is further exacerbated by non-exhaustive labelling of the training data. Therefore we conclude that instability is a necessary result of how the problem of computer vision is currently formulated. While the problem cannot be eliminated, through the analysis of the causes, we have arrived at ways how it can be partially alleviated. These include i) increasing the resolution of images, ii) providing contextual information for the image, iii) exhaustive labelling of training data, and iv) preventing attackers from frequent access to the computer vision system.



## **10. SoK: Pitfalls in Evaluating Black-Box Attacks**

cs.CR

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17534v1) [paper-pdf](http://arxiv.org/pdf/2310.17534v1)

**Authors**: Fnu Suya, Anshuman Suri, Tingwei Zhang, Jingtao Hong, Yuan Tian, David Evans

**Abstract**: Numerous works study black-box attacks on image classifiers. However, these works make different assumptions on the adversary's knowledge and current literature lacks a cohesive organization centered around the threat model. To systematize knowledge in this area, we propose a taxonomy over the threat space spanning the axes of feedback granularity, the access of interactive queries, and the quality and quantity of the auxiliary data available to the attacker. Our new taxonomy provides three key insights. 1) Despite extensive literature, numerous under-explored threat spaces exist, which cannot be trivially solved by adapting techniques from well-explored settings. We demonstrate this by establishing a new state-of-the-art in the less-studied setting of access to top-k confidence scores by adapting techniques from well-explored settings of accessing the complete confidence vector, but show how it still falls short of the more restrictive setting that only obtains the prediction label, highlighting the need for more research. 2) Identification the threat model of different attacks uncovers stronger baselines that challenge prior state-of-the-art claims. We demonstrate this by enhancing an initially weaker baseline (under interactive query access) via surrogate models, effectively overturning claims in the respective paper. 3) Our taxonomy reveals interactions between attacker knowledge that connect well to related areas, such as model inversion and extraction attacks. We discuss how advances in other areas can enable potentially stronger black-box attacks. Finally, we emphasize the need for a more realistic assessment of attack success by factoring in local attack runtime. This approach reveals the potential for certain attacks to achieve notably higher success rates and the need to evaluate attacks in diverse and harder settings, highlighting the need for better selection criteria.



## **11. CBD: A Certified Backdoor Detector Based on Local Dominant Probability**

cs.LG

Accepted to NeurIPS 2023

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17498v1) [paper-pdf](http://arxiv.org/pdf/2310.17498v1)

**Authors**: Zhen Xiang, Zidi Xiong, Bo Li

**Abstract**: Backdoor attack is a common threat to deep neural networks. During testing, samples embedded with a backdoor trigger will be misclassified as an adversarial target by a backdoored model, while samples without the backdoor trigger will be correctly classified. In this paper, we present the first certified backdoor detector (CBD), which is based on a novel, adjustable conformal prediction scheme based on our proposed statistic local dominant probability. For any classifier under inspection, CBD provides 1) a detection inference, 2) the condition under which the attacks are guaranteed to be detectable for the same classification domain, and 3) a probabilistic upper bound for the false positive rate. Our theoretical results show that attacks with triggers that are more resilient to test-time noise and have smaller perturbation magnitudes are more likely to be detected with guarantees. Moreover, we conduct extensive experiments on four benchmark datasets considering various backdoor types, such as BadNet, CB, and Blend. CBD achieves comparable or even higher detection accuracy than state-of-the-art detectors, and it in addition provides detection certification. Notably, for backdoor attacks with random perturbation triggers bounded by $\ell_2\leq0.75$ which achieves more than 90\% attack success rate, CBD achieves 100\% (98\%), 100\% (84\%), 98\% (98\%), and 72\% (40\%) empirical (certified) detection true positive rates on the four benchmark datasets GTSRB, SVHN, CIFAR-10, and TinyImageNet, respectively, with low false positive rates.



## **12. Uncertainty-weighted Loss Functions for Improved Adversarial Attacks on Semantic Segmentation**

cs.CV

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17436v1) [paper-pdf](http://arxiv.org/pdf/2310.17436v1)

**Authors**: Kira Maag, Asja Fischer

**Abstract**: State-of-the-art deep neural networks have been shown to be extremely powerful in a variety of perceptual tasks like semantic segmentation. However, these networks are vulnerable to adversarial perturbations of the input which are imperceptible for humans but lead to incorrect predictions. Treating image segmentation as a sum of pixel-wise classifications, adversarial attacks developed for classification models were shown to be applicable to segmentation models as well. In this work, we present simple uncertainty-based weighting schemes for the loss functions of such attacks that (i) put higher weights on pixel classifications which can more easily perturbed and (ii) zero-out the pixel-wise losses corresponding to those pixels that are already confidently misclassified. The weighting schemes can be easily integrated into the loss function of a range of well-known adversarial attackers with minimal additional computational overhead, but lead to significant improved perturbation performance, as we demonstrate in our empirical analysis on several datasets and models.



## **13. Stealthy SWAPs: Adversarial SWAP Injection in Multi-Tenant Quantum Computing**

quant-ph

7 pages, VLSID

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17426v1) [paper-pdf](http://arxiv.org/pdf/2310.17426v1)

**Authors**: Suryansh Upadhyay, Swaroop Ghosh

**Abstract**: Quantum computing (QC) holds tremendous promise in revolutionizing problem-solving across various domains. It has been suggested in literature that 50+ qubits are sufficient to achieve quantum advantage (i.e., to surpass supercomputers in solving certain class of optimization problems).The hardware size of existing Noisy Intermediate-Scale Quantum (NISQ) computers have been ever increasing over the years. Therefore, Multi-tenant computing (MTC) has emerged as a potential solution for efficient hardware utilization, enabling shared resource access among multiple quantum programs. However, MTC can also bring new security concerns. This paper proposes one such threat for MTC in superconducting quantum hardware i.e., adversarial SWAP gate injection in victims program during compilation for MTC. We present a representative scheduler designed for optimal resource allocation. To demonstrate the impact of this attack model, we conduct a detailed case study using a sample scheduler. Exhaustive experiments on circuits with varying depths and qubits offer valuable insights into the repercussions of these attacks. We report a max of approximately 55 percent and a median increase of approximately 25 percent in SWAP overhead. As a countermeasure, we also propose a sample machine learning model for detecting any abnormal user behavior and priority adjustment.



## **14. Detection Defenses: An Empty Promise against Adversarial Patch Attacks on Optical Flow**

cs.CV

Accepted to WACV 2024

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17403v1) [paper-pdf](http://arxiv.org/pdf/2310.17403v1)

**Authors**: Erik Scheurer, Jenny Schmalfuss, Alexander Lis, Andrés Bruhn

**Abstract**: Adversarial patches undermine the reliability of optical flow predictions when placed in arbitrary scene locations. Therefore, they pose a realistic threat to real-world motion detection and its downstream applications. Potential remedies are defense strategies that detect and remove adversarial patches, but their influence on the underlying motion prediction has not been investigated. In this paper, we thoroughly examine the currently available detect-and-remove defenses ILP and LGS for a wide selection of state-of-the-art optical flow methods, and illuminate their side effects on the quality and robustness of the final flow predictions. In particular, we implement defense-aware attacks to investigate whether current defenses are able to withstand attacks that take the defense mechanism into account. Our experiments yield two surprising results: Detect-and-remove defenses do not only lower the optical flow quality on benign scenes, in doing so, they also harm the robustness under patch attacks for all tested optical flow methods except FlowNetC. As currently employed detect-and-remove defenses fail to deliver the promised adversarial robustness for optical flow, they evoke a false sense of security. The code is available at https://github.com/cv-stuttgart/DetectionDefenses.



## **15. Learning Transferable Adversarial Robust Representations via Multi-view Consistency**

cs.LG

*Equal contribution (Author ordering determined by coin flip).  NeurIPS SafetyML workshop 2022, Under review

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2210.10485v2) [paper-pdf](http://arxiv.org/pdf/2210.10485v2)

**Authors**: Minseon Kim, Hyeonjeong Ha, Dong Bok Lee, Sung Ju Hwang

**Abstract**: Despite the success on few-shot learning problems, most meta-learned models only focus on achieving good performance on clean examples and thus easily break down when given adversarially perturbed samples. While some recent works have shown that a combination of adversarial learning and meta-learning could enhance the robustness of a meta-learner against adversarial attacks, they fail to achieve generalizable adversarial robustness to unseen domains and tasks, which is the ultimate goal of meta-learning. To address this challenge, we propose a novel meta-adversarial multi-view representation learning framework with dual encoders. Specifically, we introduce the discrepancy across the two differently augmented samples of the same data instance by first updating the encoder parameters with them and further imposing a novel label-free adversarial attack to maximize their discrepancy. Then, we maximize the consistency across the views to learn transferable robust representations across domains and tasks. Through experimental validation on multiple benchmarks, we demonstrate the effectiveness of our framework on few-shot learning tasks from unseen domains, achieving over 10\% robust accuracy improvements against previous adversarial meta-learning baselines.



## **16. Effective Targeted Attacks for Adversarial Self-Supervised Learning**

cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2210.10482v2) [paper-pdf](http://arxiv.org/pdf/2210.10482v2)

**Authors**: Minseon Kim, Hyeonjeong Ha, Sooel Son, Sung Ju Hwang

**Abstract**: Recently, unsupervised adversarial training (AT) has been highlighted as a means of achieving robustness in models without any label information. Previous studies in unsupervised AT have mostly focused on implementing self-supervised learning (SSL) frameworks, which maximize the instance-wise classification loss to generate adversarial examples. However, we observe that simply maximizing the self-supervised training loss with an untargeted adversarial attack often results in generating ineffective adversaries that may not help improve the robustness of the trained model, especially for non-contrastive SSL frameworks without negative examples. To tackle this problem, we propose a novel positive mining for targeted adversarial attack to generate effective adversaries for adversarial SSL frameworks. Specifically, we introduce an algorithm that selects the most confusing yet similar target example for a given instance based on entropy and similarity, and subsequently perturbs the given instance towards the selected target. Our method demonstrates significant enhancements in robustness when applied to non-contrastive SSL frameworks, and less but consistent robustness improvements with contrastive SSL frameworks, on the benchmark datasets.



## **17. Efficient Adversarial Contrastive Learning via Robustness-Aware Coreset Selection**

cs.LG

NeurIPS 2023 Spotlight

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2302.03857v5) [paper-pdf](http://arxiv.org/pdf/2302.03857v5)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL) does not require expensive data annotations but outputs a robust representation that withstands adversarial attacks and also generalizes to a wide range of downstream tasks. However, ACL needs tremendous running time to generate the adversarial variants of all training data, which limits its scalability to large datasets. To speed up ACL, this paper proposes a robustness-aware coreset selection (RCS) method. RCS does not require label information and searches for an informative subset that minimizes a representational divergence, which is the distance of the representation between natural data and their virtual adversarial variants. The vanilla solution of RCS via traversing all possible subsets is computationally prohibitive. Therefore, we theoretically transform RCS into a surrogate problem of submodular maximization, of which the greedy search is an efficient solution with an optimality guarantee for the original problem. Empirically, our comprehensive results corroborate that RCS can speed up ACL by a large margin without significantly hurting the robustness transferability. Notably, to the best of our knowledge, we are the first to conduct ACL efficiently on the large-scale ImageNet-1K dataset to obtain an effective robust representation via RCS. Our source code is at https://github.com/GodXuxilie/Efficient_ACL_via_RCS.



## **18. Detecting stealthy cyberattacks on adaptive cruise control vehicles: A machine learning approach**

cs.MA

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17091v1) [paper-pdf](http://arxiv.org/pdf/2310.17091v1)

**Authors**: Tianyi Li, Mingfeng Shang, Shian Wang, Raphael Stern

**Abstract**: With the advent of vehicles equipped with advanced driver-assistance systems, such as adaptive cruise control (ACC) and other automated driving features, the potential for cyberattacks on these automated vehicles (AVs) has emerged. While overt attacks that force vehicles to collide may be easily identified, more insidious attacks, which only slightly alter driving behavior, can result in network-wide increases in congestion, fuel consumption, and even crash risk without being easily detected. To address the detection of such attacks, we first present a traffic model framework for three types of potential cyberattacks: malicious manipulation of vehicle control commands, false data injection attacks on sensor measurements, and denial-of-service (DoS) attacks. We then investigate the impacts of these attacks at both the individual vehicle (micro) and traffic flow (macro) levels. A novel generative adversarial network (GAN)-based anomaly detection model is proposed for real-time identification of such attacks using vehicle trajectory data. We provide numerical evidence {to demonstrate} the efficacy of our machine learning approach in detecting cyberattacks on ACC-equipped vehicles. The proposed method is compared against some recently proposed neural network models and observed to have higher accuracy in identifying anomalous driving behaviors of ACC vehicles.



## **19. Trust, but Verify: Robust Image Segmentation using Deep Learning**

cs.CV

5 Pages, 8 Figures, conference

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2310.16999v1) [paper-pdf](http://arxiv.org/pdf/2310.16999v1)

**Authors**: Fahim Ahmed Zaman, Xiaodong Wu, Weiyu Xu, Milan Sonka, Raghuraman Mudumbai

**Abstract**: We describe a method for verifying the output of a deep neural network for medical image segmentation that is robust to several classes of random as well as worst-case perturbations i.e. adversarial attacks. This method is based on a general approach recently developed by the authors called ``Trust, but Verify" wherein an auxiliary verification network produces predictions about certain masked features in the input image using the segmentation as an input. A well-designed auxiliary network will produce high-quality predictions when the input segmentations are accurate, but will produce low-quality predictions when the segmentations are incorrect. Checking the predictions of such a network with the original image allows us to detect bad segmentations. However, to ensure the verification method is truly robust, we need a method for checking the quality of the predictions that does not itself rely on a black-box neural network. Indeed, we show that previous methods for segmentation evaluation that do use deep neural regression networks are vulnerable to false negatives i.e. can inaccurately label bad segmentations as good. We describe the design of a verification network that avoids such vulnerability and present results to demonstrate its robustness compared to previous methods.



## **20. Break it, Imitate it, Fix it: Robustness by Generating Human-Like Attacks**

cs.LG

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2310.16955v1) [paper-pdf](http://arxiv.org/pdf/2310.16955v1)

**Authors**: Aradhana Sinha, Ananth Balashankar, Ahmad Beirami, Thi Avrahami, Jilin Chen, Alex Beutel

**Abstract**: Real-world natural language processing systems need to be robust to human adversaries. Collecting examples of human adversaries for training is an effective but expensive solution. On the other hand, training on synthetic attacks with small perturbations - such as word-substitution - does not actually improve robustness to human adversaries. In this paper, we propose an adversarial training framework that uses limited human adversarial examples to generate more useful adversarial examples at scale. We demonstrate the advantages of this system on the ANLI and hate speech detection benchmark datasets - both collected via an iterative, adversarial human-and-model-in-the-loop procedure. Compared to training only on observed human attacks, also training on our synthetic adversarial examples improves model robustness to future rounds. In ANLI, we see accuracy gains on the current set of attacks (44.1%$\,\to\,$50.1%) and on two future unseen rounds of human generated attacks (32.5%$\,\to\,$43.4%, and 29.4%$\,\to\,$40.2%). In hate speech detection, we see AUC gains on current attacks (0.76 $\to$ 0.84) and a future round (0.77 $\to$ 0.79). Attacks from methods that do not learn the distribution of existing human adversaries, meanwhile, degrade robustness.



## **21. A Vulnerability of Attribution Methods Using Pre-Softmax Scores**

cs.LG

7 pages, 5 figures

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2307.03305v2) [paper-pdf](http://arxiv.org/pdf/2307.03305v2)

**Authors**: Miguel Lerma, Mirtha Lucas

**Abstract**: We discuss a vulnerability involving a category of attribution methods used to provide explanations for the outputs of convolutional neural networks working as classifiers. It is known that this type of networks are vulnerable to adversarial attacks, in which imperceptible perturbations of the input may alter the outputs of the model. In contrast, here we focus on effects that small modifications in the model may cause on the attribution method without altering the model outputs.



## **22. Dual Defense: Adversarial, Traceable, and Invisible Robust Watermarking against Face Swapping**

cs.CV

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2310.16540v1) [paper-pdf](http://arxiv.org/pdf/2310.16540v1)

**Authors**: Yunming Zhang, Dengpan Ye, Caiyun Xie, Long Tang, Chuanxi Chen, Ziyi Liu, Jiacheng Deng

**Abstract**: The malicious applications of deep forgery, represented by face swapping, have introduced security threats such as misinformation dissemination and identity fraud. While some research has proposed the use of robust watermarking methods to trace the copyright of facial images for post-event traceability, these methods cannot effectively prevent the generation of forgeries at the source and curb their dissemination. To address this problem, we propose a novel comprehensive active defense mechanism that combines traceability and adversariality, called Dual Defense. Dual Defense invisibly embeds a single robust watermark within the target face to actively respond to sudden cases of malicious face swapping. It disrupts the output of the face swapping model while maintaining the integrity of watermark information throughout the entire dissemination process. This allows for watermark extraction at any stage of image tracking for traceability. Specifically, we introduce a watermark embedding network based on original-domain feature impersonation attack. This network learns robust adversarial features of target facial images and embeds watermarks, seeking a well-balanced trade-off between watermark invisibility, adversariality, and traceability through perceptual adversarial encoding strategies. Extensive experiments demonstrate that Dual Defense achieves optimal overall defense success rates and exhibits promising universality in anti-face swapping tasks and dataset generalization ability. It maintains impressive adversariality and traceability in both original and robust settings, surpassing current forgery defense methods that possess only one of these capabilities, including CMUA-Watermark, Anti-Forgery, FakeTagger, or PGD methods.



## **23. Universal adversarial perturbations for multiple classification tasks with quantum classifiers**

quant-ph

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2306.11974v3) [paper-pdf](http://arxiv.org/pdf/2306.11974v3)

**Authors**: Yun-Zhong Qiu

**Abstract**: Quantum adversarial machine learning is an emerging field that studies the vulnerability of quantum learning systems against adversarial perturbations and develops possible defense strategies. Quantum universal adversarial perturbations are small perturbations, which can make different input samples into adversarial examples that may deceive a given quantum classifier. This is a field that was rarely looked into but worthwhile investigating because universal perturbations might simplify malicious attacks to a large extent, causing unexpected devastation to quantum machine learning models. In this paper, we take a step forward and explore the quantum universal perturbations in the context of heterogeneous classification tasks. In particular, we find that quantum classifiers that achieve almost state-of-the-art accuracy on two different classification tasks can be both conclusively deceived by one carefully-crafted universal perturbation. This result is explicitly demonstrated with well-designed quantum continual learning models with elastic weight consolidation method to avoid catastrophic forgetting, as well as real-life heterogeneous datasets from hand-written digits and medical MRI images. Our results provide a simple and efficient way to generate universal perturbations on heterogeneous classification tasks and thus would provide valuable guidance for future quantum learning technologies.



## **24. Impartial Games: A Challenge for Reinforcement Learning**

cs.LG

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2205.12787v2) [paper-pdf](http://arxiv.org/pdf/2205.12787v2)

**Authors**: Bei Zhou, Søren Riis

**Abstract**: AlphaZero-style reinforcement learning (RL) algorithms excel in various board games but face challenges with impartial games, where players share pieces. We present a concrete example of a game - namely the children's game of nim - and other impartial games that seem to be a stumbling block for AlphaZero-style and similar reinforcement learning algorithms.   Our findings are consistent with recent studies showing that AlphaZero-style algorithms are vulnerable to adversarial attacks and adversarial perturbations, showing the difficulty of learning to master the games in all legal states.   We show that nim can be learned on small boards, but AlphaZero-style algorithms learning dramatically slows down when the board size increases. Intuitively, the difference between impartial games like nim and partisan games like Chess and Go can be explained by the fact that if a tiny amount of noise is added to the system (e.g. if a small part of the board is covered), for impartial games, it is typically not possible to predict whether the position is good or bad (won or lost). There is often zero correlation between the visible part of a partly blanked-out position and its correct evaluation. This situation starkly contrasts partisan games where a partly blanked-out configuration typically provides abundant or at least non-trifle information about the value of the fully uncovered position.



## **25. Defense Against Model Extraction Attacks on Recommender Systems**

cs.LG

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2310.16335v1) [paper-pdf](http://arxiv.org/pdf/2310.16335v1)

**Authors**: Sixiao Zhang, Hongzhi Yin, Hongxu Chen, Cheng Long

**Abstract**: The robustness of recommender systems has become a prominent topic within the research community. Numerous adversarial attacks have been proposed, but most of them rely on extensive prior knowledge, such as all the white-box attacks or most of the black-box attacks which assume that certain external knowledge is available. Among these attacks, the model extraction attack stands out as a promising and practical method, involving training a surrogate model by repeatedly querying the target model. However, there is a significant gap in the existing literature when it comes to defending against model extraction attacks on recommender systems. In this paper, we introduce Gradient-based Ranking Optimization (GRO), which is the first defense strategy designed to counter such attacks. We formalize the defense as an optimization problem, aiming to minimize the loss of the protected target model while maximizing the loss of the attacker's surrogate model. Since top-k ranking lists are non-differentiable, we transform them into swap matrices which are instead differentiable. These swap matrices serve as input to a student model that emulates the surrogate model's behavior. By back-propagating the loss of the student model, we obtain gradients for the swap matrices. These gradients are used to compute a swap loss, which maximizes the loss of the student model. We conducted experiments on three benchmark datasets to evaluate the performance of GRO, and the results demonstrate its superior effectiveness in defending against model extraction attacks.



## **26. UPTON: Preventing Authorship Leakage from Public Text Release via Data Poisoning**

cs.CY

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2211.09717v3) [paper-pdf](http://arxiv.org/pdf/2211.09717v3)

**Authors**: Ziyao Wang, Thai Le, Dongwon Lee

**Abstract**: Consider a scenario where an author-e.g., activist, whistle-blower, with many public writings wishes to write "anonymously" when attackers may have already built an authorship attribution (AA) model based off of public writings including those of the author. To enable her wish, we ask a question "Can one make the publicly released writings, T, unattributable so that AA models trained on T cannot attribute its authorship well?" Toward this question, we present a novel solution, UPTON, that exploits black-box data poisoning methods to weaken the authorship features in training samples and make released texts unlearnable. It is different from previous obfuscation works-e.g., adversarial attacks that modify test samples or backdoor works that only change the model outputs when triggering words occur. Using four authorship datasets (IMDb10, IMDb64, Enron, and WJO), we present empirical validation where UPTON successfully downgrades the accuracy of AA models to the impractical level (~35%) while keeping texts still readable (semantic similarity>0.9). UPTON remains effective to AA models that are already trained on available clean writings of authors.



## **27. Database Matching Under Noisy Synchronization Errors**

cs.IT

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2301.06796v2) [paper-pdf](http://arxiv.org/pdf/2301.06796v2)

**Authors**: Serhat Bakirtas, Elza Erkip

**Abstract**: The re-identification or de-anonymization of users from anonymized data through matching with publicly available correlated user data has raised privacy concerns, leading to the complementary measure of obfuscation in addition to anonymization. Recent research provides a fundamental understanding of the conditions under which privacy attacks, in the form of database matching, are successful in the presence of obfuscation. Motivated by synchronization errors stemming from the sampling of time-indexed databases, this paper presents a unified framework considering both obfuscation and synchronization errors and investigates the matching of databases under noisy entry repetitions. By investigating different structures for the repetition pattern, replica detection and seeded deletion detection algorithms are devised and sufficient and necessary conditions for successful matching are derived. Finally, the impacts of some variations of the underlying assumptions, such as the adversarial deletion model, seedless database matching, and zero-rate regime, on the results are discussed. Overall, our results provide insights into the privacy-preserving publication of anonymized and obfuscated time-indexed data as well as the closely related problem of the capacity of synchronization channels.



## **28. A Generative Framework for Low-Cost Result Validation of Outsourced Machine Learning Tasks**

cs.CR

15 pages, 11 figures

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2304.00083v2) [paper-pdf](http://arxiv.org/pdf/2304.00083v2)

**Authors**: Abhinav Kumar, Miguel A. Guirao Aguilera, Reza Tourani, Satyajayant Misra

**Abstract**: The growing popularity of Machine Learning (ML) has led to its deployment in various sensitive domains, which has resulted in significant research focused on ML security and privacy. However, in some applications, such as autonomous driving, integrity verification of the outsourced ML workload is more critical--a facet that has not received much attention. Existing solutions, such as multi-party computation and proof-based systems, impose significant computation overhead, which makes them unfit for real-time applications. We propose Fides, a novel framework for real-time validation of outsourced ML workloads. Fides features a novel and efficient distillation technique--Greedy Distillation Transfer Learning--that dynamically distills and fine-tunes a space and compute-efficient verification model for verifying the corresponding service model while running inside a trusted execution environment. Fides features a client-side attack detection model that uses statistical analysis and divergence measurements to identify, with a high likelihood, if the service model is under attack. Fides also offers a re-classification functionality that predicts the original class whenever an attack is identified. We devised a generative adversarial network framework for training the attack detection and re-classification models. The evaluation shows that Fides achieves an accuracy of up to 98% for attack detection and 94% for re-classification.



## **29. FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering**

cs.CR

22 pages (including bibliography and Appendix), Submitted to USENIX  Security '24

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.16152v1) [paper-pdf](http://arxiv.org/pdf/2310.16152v1)

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Kang Gu, Najrin Sultana, Shagufta Mehnaz

**Abstract**: Federated learning (FL) is becoming a key component in many technology-based applications including language modeling -- where individual FL participants often have privacy-sensitive text data in their local datasets. However, realizing the extent of privacy leakage in federated language models is not straightforward and the existing attacks only intend to extract data regardless of how sensitive or naive it is. To fill this gap, in this paper, we introduce two novel findings with regard to leaking privacy-sensitive user data from federated language models. Firstly, we make a key observation that model snapshots from the intermediate rounds in FL can cause greater privacy leakage than the final trained model. Secondly, we identify that privacy leakage can be aggravated by tampering with a model's selective weights that are specifically responsible for memorizing the sensitive training data. We show how a malicious client can leak the privacy-sensitive data of some other user in FL even without any cooperation from the server. Our best-performing method improves the membership inference recall by 29% and achieves up to 70% private data reconstruction, evidently outperforming existing attacks with stronger assumptions of adversary capabilities.



## **30. Diffusion-Based Adversarial Purification for Speaker Verification**

eess.AS

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.14270v2) [paper-pdf](http://arxiv.org/pdf/2310.14270v2)

**Authors**: Yibo Bai, Xiao-Lei Zhang

**Abstract**: Recently, automatic speaker verification (ASV) based on deep learning is easily contaminated by adversarial attacks, which is a new type of attack that injects imperceptible perturbations to audio signals so as to make ASV produce wrong decisions. This poses a significant threat to the security and reliability of ASV systems. To address this issue, we propose a Diffusion-Based Adversarial Purification (DAP) method that enhances the robustness of ASV systems against such adversarial attacks. Our method leverages a conditional denoising diffusion probabilistic model to effectively purify the adversarial examples and mitigate the impact of perturbations. DAP first introduces controlled noise into adversarial examples, and then performs a reverse denoising process to reconstruct clean audio. Experimental results demonstrate the efficacy of the proposed DAP in enhancing the security of ASV and meanwhile minimizing the distortion of the purified audio signals.



## **31. A Survey on LLM-generated Text Detection: Necessity, Methods, and Future Directions**

cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.14724v2) [paper-pdf](http://arxiv.org/pdf/2310.14724v2)

**Authors**: Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Derek F. Wong, Lidia S. Chao

**Abstract**: The powerful ability to understand, follow, and generate complex language emerging from large language models (LLMs) makes LLM-generated text flood many areas of our daily lives at an incredible speed and is widely accepted by humans. As LLMs continue to expand, there is an imperative need to develop detectors that can detect LLM-generated text. This is crucial to mitigate potential misuse of LLMs and safeguard realms like artistic expression and social networks from harmful influence of LLM-generated content. The LLM-generated text detection aims to discern if a piece of text was produced by an LLM, which is essentially a binary classification task. The detector techniques have witnessed notable advancements recently, propelled by innovations in watermarking techniques, zero-shot methods, fine-turning LMs methods, adversarial learning methods, LLMs as detectors, and human-assisted methods. In this survey, we collate recent research breakthroughs in this area and underscore the pressing need to bolster detector research. We also delve into prevalent datasets, elucidating their limitations and developmental requirements. Furthermore, we analyze various LLM-generated text detection paradigms, shedding light on challenges like out-of-distribution problems, potential attacks, and data ambiguity. Conclusively, we highlight interesting directions for future research in LLM-generated text detection to advance the implementation of responsible artificial intelligence (AI). Our aim with this survey is to provide a clear and comprehensive introduction for newcomers while also offering seasoned researchers a valuable update in the field of LLM-generated text detection. The useful resources are publicly available at: https://github.com/NLP2CT/LLM-generated-Text-Detection.



## **32. Momentum Gradient-based Untargeted Attack on Hypergraph Neural Networks**

cs.LG

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15656v1) [paper-pdf](http://arxiv.org/pdf/2310.15656v1)

**Authors**: Yang Chen, Stjepan Picek, Zhonglin Ye, Zhaoyang Wang, Haixing Zhao

**Abstract**: Hypergraph Neural Networks (HGNNs) have been successfully applied in various hypergraph-related tasks due to their excellent higher-order representation capabilities. Recent works have shown that deep learning models are vulnerable to adversarial attacks. Most studies on graph adversarial attacks have focused on Graph Neural Networks (GNNs), and the study of adversarial attacks on HGNNs remains largely unexplored. In this paper, we try to reduce this gap. We design a new HGNNs attack model for the untargeted attack, namely MGHGA, which focuses on modifying node features. We consider the process of HGNNs training and use a surrogate model to implement the attack before hypergraph modeling. Specifically, MGHGA consists of two parts: feature selection and feature modification. We use a momentum gradient mechanism to choose the attack node features in the feature selection module. In the feature modification module, we use two feature generation approaches (direct modification and sign gradient) to enable MGHGA to be employed on discrete and continuous datasets. We conduct extensive experiments on five benchmark datasets to validate the attack performance of MGHGA in the node and the visual object classification tasks. The results show that MGHGA improves performance by an average of 2% compared to the than the baselines.



## **33. Deceptive Fairness Attacks on Graphs via Meta Learning**

cs.LG

23 pages, 11 tables

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15653v1) [paper-pdf](http://arxiv.org/pdf/2310.15653v1)

**Authors**: Jian Kang, Yinglong Xia, Ross Maciejewski, Jiebo Luo, Hanghang Tong

**Abstract**: We study deceptive fairness attacks on graphs to answer the following question: How can we achieve poisoning attacks on a graph learning model to exacerbate the bias deceptively? We answer this question via a bi-level optimization problem and propose a meta learning-based framework named FATE. FATE is broadly applicable with respect to various fairness definitions and graph learning models, as well as arbitrary choices of manipulation operations. We further instantiate FATE to attack statistical parity and individual fairness on graph neural networks. We conduct extensive experimental evaluations on real-world datasets in the task of semi-supervised node classification. The experimental results demonstrate that FATE could amplify the bias of graph neural networks with or without fairness consideration while maintaining the utility on the downstream task. We hope this paper provides insights into the adversarial robustness of fair graph learning and can shed light on designing robust and fair graph learning in future studies.



## **34. Facial Data Minimization: Shallow Model as Your Privacy Filter**

cs.CR

14 pages, 11 figures

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15590v1) [paper-pdf](http://arxiv.org/pdf/2310.15590v1)

**Authors**: Yuwen Pu, Jiahao Chen, Jiayu Pan, Hao li, Diqun Yan, Xuhong Zhang, Shouling Ji

**Abstract**: Face recognition service has been used in many fields and brings much convenience to people. However, once the user's facial data is transmitted to a service provider, the user will lose control of his/her private data. In recent years, there exist various security and privacy issues due to the leakage of facial data. Although many privacy-preserving methods have been proposed, they usually fail when they are not accessible to adversaries' strategies or auxiliary data. Hence, in this paper, by fully considering two cases of uploading facial images and facial features, which are very typical in face recognition service systems, we proposed a data privacy minimization transformation (PMT) method. This method can process the original facial data based on the shallow model of authorized services to obtain the obfuscated data. The obfuscated data can not only maintain satisfactory performance on authorized models and restrict the performance on other unauthorized models but also prevent original privacy data from leaking by AI methods and human visual theft. Additionally, since a service provider may execute preprocessing operations on the received data, we also propose an enhanced perturbation method to improve the robustness of PMT. Besides, to authorize one facial image to multiple service models simultaneously, a multiple restriction mechanism is proposed to improve the scalability of PMT. Finally, we conduct extensive experiments and evaluate the effectiveness of the proposed PMT in defending against face reconstruction, data abuse, and face attribute estimation attacks. These experimental results demonstrate that PMT performs well in preventing facial data abuse and privacy leakage while maintaining face recognition accuracy.



## **35. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2308.07308v3) [paper-pdf](http://arxiv.org/pdf/2308.07308v3)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2.



## **36. Fast Propagation is Better: Accelerating Single-Step Adversarial Training via Sampling Subnetworks**

cs.CV

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15444v1) [paper-pdf](http://arxiv.org/pdf/2310.15444v1)

**Authors**: Xiaojun Jia, Jianshu Li, Jindong Gu, Yang Bai, Xiaochun Cao

**Abstract**: Adversarial training has shown promise in building robust models against adversarial examples. A major drawback of adversarial training is the computational overhead introduced by the generation of adversarial examples. To overcome this limitation, adversarial training based on single-step attacks has been explored. Previous work improves the single-step adversarial training from different perspectives, e.g., sample initialization, loss regularization, and training strategy. Almost all of them treat the underlying model as a black box. In this work, we propose to exploit the interior building blocks of the model to improve efficiency. Specifically, we propose to dynamically sample lightweight subnetworks as a surrogate model during training. By doing this, both the forward and backward passes can be accelerated for efficient adversarial training. Besides, we provide theoretical analysis to show the model robustness can be improved by the single-step adversarial training with sampled subnetworks. Furthermore, we propose a novel sampling strategy where the sampling varies from layer to layer and from iteration to iteration. Compared with previous methods, our method not only reduces the training cost but also achieves better model robustness. Evaluations on a series of popular datasets demonstrate the effectiveness of the proposed FB-Better. Our code has been released at https://github.com/jiaxiaojunQAQ/FP-Better.



## **37. Unsupervised Federated Learning: A Federated Gradient EM Algorithm for Heterogeneous Mixture Models with Robustness against Adversarial Attacks**

stat.ML

43 pages, 1 figure

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15330v1) [paper-pdf](http://arxiv.org/pdf/2310.15330v1)

**Authors**: Ye Tian, Haolei Weng, Yang Feng

**Abstract**: While supervised federated learning approaches have enjoyed significant success, the domain of unsupervised federated learning remains relatively underexplored. In this paper, we introduce a novel federated gradient EM algorithm designed for the unsupervised learning of mixture models with heterogeneous mixture proportions across tasks. We begin with a comprehensive finite-sample theory that holds for general mixture models, then apply this general theory on Gaussian Mixture Models (GMMs) and Mixture of Regressions (MoRs) to characterize the explicit estimation error of model parameters and mixture proportions. Our proposed federated gradient EM algorithm demonstrates several key advantages: adaptability to unknown task similarity, resilience against adversarial attacks on a small fraction of data sources, protection of local data privacy, and computational and communication efficiency.



## **38. GRASP: Accelerating Shortest Path Attacks via Graph Attention**

cs.LG

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.07980v2) [paper-pdf](http://arxiv.org/pdf/2310.07980v2)

**Authors**: Zohair Shafi, Benjamin A. Miller, Ayan Chatterjee, Tina Eliassi-Rad, Rajmonda S. Caceres

**Abstract**: Recent advances in machine learning (ML) have shown promise in aiding and accelerating classical combinatorial optimization algorithms. ML-based speed ups that aim to learn in an end to end manner (i.e., directly output the solution) tend to trade off run time with solution quality. Therefore, solutions that are able to accelerate existing solvers while maintaining their performance guarantees, are of great interest. We consider an APX-hard problem, where an adversary aims to attack shortest paths in a graph by removing the minimum number of edges. We propose the GRASP algorithm: Graph Attention Accelerated Shortest Path Attack, an ML aided optimization algorithm that achieves run times up to 10x faster, while maintaining the quality of solution generated. GRASP uses a graph attention network to identify a smaller subgraph containing the combinatorial solution, thus effectively reducing the input problem size. Additionally, we demonstrate how careful representation of the input graph, including node features that correlate well with the optimization task, can highlight important structure in the optimization solution.



## **39. AutoDAN: Automatic and Interpretable Adversarial Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15140v1) [paper-pdf](http://arxiv.org/pdf/2310.15140v1)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent work suggests that patching LLMs against these attacks is possible: manual jailbreak attacks are human-readable but often limited and public, making them easy to block; adversarial attacks generate gibberish prompts that can be detected using perplexity-based filters. In this paper, we show that these solutions may be too optimistic. We propose an interpretable adversarial attack, \texttt{AutoDAN}, that combines the strengths of both types of attacks. It automatically generates attack prompts that bypass perplexity-based filters while maintaining a high attack success rate like manual jailbreak attacks. These prompts are interpretable and diverse, exhibiting strategies commonly used in manual jailbreak attacks, and transfer better than their non-readable counterparts when using limited training data or a single proxy model. We also customize \texttt{AutoDAN}'s objective to leak system prompts, another jailbreak application not addressed in the adversarial attack literature. Our work provides a new way to red-team LLMs and to understand the mechanism of jailbreak attacks.



## **40. On the Detection of Image-Scaling Attacks in Machine Learning**

cs.CR

Accepted at ACSAC'23

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15085v1) [paper-pdf](http://arxiv.org/pdf/2310.15085v1)

**Authors**: Erwin Quiring, Andreas Müller, Konrad Rieck

**Abstract**: Image scaling is an integral part of machine learning and computer vision systems. Unfortunately, this preprocessing step is vulnerable to so-called image-scaling attacks where an attacker makes unnoticeable changes to an image so that it becomes a new image after scaling. This opens up new ways for attackers to control the prediction or to improve poisoning and backdoor attacks. While effective techniques exist to prevent scaling attacks, their detection has not been rigorously studied yet. Consequently, it is currently not possible to reliably spot these attacks in practice.   This paper presents the first in-depth systematization and analysis of detection methods for image-scaling attacks. We identify two general detection paradigms and derive novel methods from them that are simple in design yet significantly outperform previous work. We demonstrate the efficacy of these methods in a comprehensive evaluation with all major learning platforms and scaling algorithms. First, we show that image-scaling attacks modifying the entire scaled image can be reliably detected even under an adaptive adversary. Second, we find that our methods provide strong detection performance even if only minor parts of the image are manipulated. As a result, we can introduce a novel protection layer against image-scaling attacks.



## **41. Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization**

cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2305.00374v2) [paper-pdf](http://arxiv.org/pdf/2305.00374v2)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL) is a technique that enhances standard contrastive learning (SCL) by incorporating adversarial data to learn a robust representation that can withstand adversarial attacks and common corruptions without requiring costly annotations. To improve transferability, the existing work introduced the standard invariant regularization (SIR) to impose style-independence property to SCL, which can exempt the impact of nuisance style factors in the standard representation. However, it is unclear how the style-independence property benefits ACL-learned robust representations. In this paper, we leverage the technique of causal reasoning to interpret the ACL and propose adversarial invariant regularization (AIR) to enforce independence from style factors. We regulate the ACL using both SIR and AIR to output the robust representation. Theoretically, we show that AIR implicitly encourages the representational distance between different views of natural data and their adversarial variants to be independent of style factors. Empirically, our experimental results show that invariant regularization significantly improves the performance of state-of-the-art ACL methods in terms of both standard generalization and robustness on downstream tasks. To the best of our knowledge, we are the first to apply causal reasoning to interpret ACL and develop AIR for enhancing ACL-learned robust representations. Our source code is at https://github.com/GodXuxilie/Enhancing_ACL_via_AIR.



## **42. Kidnapping Deep Learning-based Multirotors using Optimized Flying Adversarial Patches**

cs.RO

Accepted at MRS 2023, 7 pages, 5 figures. arXiv admin note:  substantial text overlap with arXiv:2305.12859

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2308.00344v2) [paper-pdf](http://arxiv.org/pdf/2308.00344v2)

**Authors**: Pia Hanfeld, Khaled Wahba, Marina M. -C. Höhne, Michael Bussmann, Wolfgang Hönig

**Abstract**: Autonomous flying robots, such as multirotors, often rely on deep learning models that make predictions based on a camera image, e.g. for pose estimation. These models can predict surprising results if applied to input images outside the training domain. This fault can be exploited by adversarial attacks, for example, by computing small images, so-called adversarial patches, that can be placed in the environment to manipulate the neural network's prediction. We introduce flying adversarial patches, where multiple images are mounted on at least one other flying robot and therefore can be placed anywhere in the field of view of a victim multirotor. By introducing the attacker robots, the system is extended to an adversarial multi-robot system. For an effective attack, we compare three methods that simultaneously optimize multiple adversarial patches and their position in the input image. We show that our methods scale well with the number of adversarial patches. Moreover, we demonstrate physical flights with two robots, where we employ a novel attack policy that uses the computed adversarial patches to kidnap a robot that was supposed to follow a human.



## **43. Beyond Hard Samples: Robust and Effective Grammatical Error Correction with Cycle Self-Augmenting**

cs.CL

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.13321v2) [paper-pdf](http://arxiv.org/pdf/2310.13321v2)

**Authors**: Zecheng Tang, Kaifeng Qi, Juntao Li, Min Zhang

**Abstract**: Recent studies have revealed that grammatical error correction methods in the sequence-to-sequence paradigm are vulnerable to adversarial attack, and simply utilizing adversarial examples in the pre-training or post-training process can significantly enhance the robustness of GEC models to certain types of attack without suffering too much performance loss on clean data. In this paper, we further conduct a thorough robustness evaluation of cutting-edge GEC methods for four different types of adversarial attacks and propose a simple yet very effective Cycle Self-Augmenting (CSA) method accordingly. By leveraging the augmenting data from the GEC models themselves in the post-training process and introducing regularization data for cycle training, our proposed method can effectively improve the model robustness of well-trained GEC models with only a few more training epochs as an extra cost. More concretely, further training on the regularization data can prevent the GEC models from over-fitting on easy-to-learn samples and thus can improve the generalization capability and robustness towards unseen data (adversarial noise/samples). Meanwhile, the self-augmented data can provide more high-quality pseudo pairs to improve model performance on the original testing data. Experiments on four benchmark datasets and seven strong models indicate that our proposed training method can significantly enhance the robustness of four types of attacks without using purposely built adversarial examples in training. Evaluation results on clean data further confirm that our proposed CSA method significantly improves the performance of four baselines and yields nearly comparable results with other state-of-the-art models. Our code is available at https://github.com/ZetangForward/CSA-GEC.



## **44. Semantic-Aware Adversarial Training for Reliable Deep Hashing Retrieval**

cs.CV

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.14637v1) [paper-pdf](http://arxiv.org/pdf/2310.14637v1)

**Authors**: Xu Yuan, Zheng Zhang, Xunguang Wang, Lin Wu

**Abstract**: Deep hashing has been intensively studied and successfully applied in large-scale image retrieval systems due to its efficiency and effectiveness. Recent studies have recognized that the existence of adversarial examples poses a security threat to deep hashing models, that is, adversarial vulnerability. Notably, it is challenging to efficiently distill reliable semantic representatives for deep hashing to guide adversarial learning, and thereby it hinders the enhancement of adversarial robustness of deep hashing-based retrieval models. Moreover, current researches on adversarial training for deep hashing are hard to be formalized into a unified minimax structure. In this paper, we explore Semantic-Aware Adversarial Training (SAAT) for improving the adversarial robustness of deep hashing models. Specifically, we conceive a discriminative mainstay features learning (DMFL) scheme to construct semantic representatives for guiding adversarial learning in deep hashing. Particularly, our DMFL with the strict theoretical guarantee is adaptively optimized in a discriminative learning manner, where both discriminative and semantic properties are jointly considered. Moreover, adversarial examples are fabricated by maximizing the Hamming distance between the hash codes of adversarial samples and mainstay features, the efficacy of which is validated in the adversarial attack trials. Further, we, for the first time, formulate the formalized adversarial training of deep hashing into a unified minimax optimization under the guidance of the generated mainstay codes. Extensive experiments on benchmark datasets show superb attack performance against the state-of-the-art algorithms, meanwhile, the proposed adversarial training can effectively eliminate adversarial perturbations for trustworthy deep hashing-based retrieval. Our code is available at https://github.com/xandery-geek/SAAT.



## **45. TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models**

cs.CR

Accepted by NeurIPS'23

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2306.06815v2) [paper-pdf](http://arxiv.org/pdf/2306.06815v2)

**Authors**: Jiaqi Xue, Mengxin Zheng, Ting Hua, Yilin Shen, Yepeng Liu, Ladislau Boloni, Qian Lou

**Abstract**: Large Language Models (LLMs) are progressively being utilized as machine learning services and interface tools for various applications. However, the security implications of LLMs, particularly in relation to adversarial and Trojan attacks, remain insufficiently examined. In this paper, we propose TrojLLM, an automatic and black-box framework to effectively generate universal and stealthy triggers. When these triggers are incorporated into the input data, the LLMs' outputs can be maliciously manipulated. Moreover, the framework also supports embedding Trojans within discrete prompts, enhancing the overall effectiveness and precision of the triggers' attacks. Specifically, we propose a trigger discovery algorithm for generating universal triggers for various inputs by querying victim LLM-based APIs using few-shot data samples. Furthermore, we introduce a novel progressive Trojan poisoning algorithm designed to generate poisoned prompts that retain efficacy and transferability across a diverse range of models. Our experiments and results demonstrate TrojLLM's capacity to effectively insert Trojans into text prompts in real-world black-box LLM APIs including GPT-3.5 and GPT-4, while maintaining exceptional performance on clean test sets. Our work sheds light on the potential security risks in current models and offers a potential defensive approach. The source code of TrojLLM is available at https://github.com/UCF-ML-Research/TrojLLM.



## **46. ADoPT: LiDAR Spoofing Attack Detection Based on Point-Level Temporal Consistency**

cs.CV

BMVC 2023 (17 pages, 13 figures, and 1 table)

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.14504v1) [paper-pdf](http://arxiv.org/pdf/2310.14504v1)

**Authors**: Minkyoung Cho, Yulong Cao, Zixiang Zhou, Z. Morley Mao

**Abstract**: Deep neural networks (DNNs) are increasingly integrated into LiDAR (Light Detection and Ranging)-based perception systems for autonomous vehicles (AVs), requiring robust performance under adversarial conditions. We aim to address the challenge of LiDAR spoofing attacks, where attackers inject fake objects into LiDAR data and fool AVs to misinterpret their environment and make erroneous decisions. However, current defense algorithms predominantly depend on perception outputs (i.e., bounding boxes) thus face limitations in detecting attackers given the bounding boxes are generated by imperfect perception models processing limited points, acquired based on the ego vehicle's viewpoint. To overcome these limitations, we propose a novel framework, named ADoPT (Anomaly Detection based on Point-level Temporal consistency), which quantitatively measures temporal consistency across consecutive frames and identifies abnormal objects based on the coherency of point clusters. In our evaluation using the nuScenes dataset, our algorithm effectively counters various LiDAR spoofing attacks, achieving a low (< 10%) false positive ratio (FPR) and high (> 85%) true positive ratio (TPR), outperforming existing state-of-the-art defense methods, CARLO and 3D-TC2. Furthermore, our evaluation demonstrates the promising potential for accurate attack detection across various road environments.



## **47. Probabilistic and Semantic Descriptions of Image Manifolds and Their Applications**

cs.CV

26 pages, 17 figures, 1 table

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2307.02881v4) [paper-pdf](http://arxiv.org/pdf/2307.02881v4)

**Authors**: Peter Tu, Zhaoyuan Yang, Richard Hartley, Zhiwei Xu, Jing Zhang, Yiwei Fu, Dylan Campbell, Jaskirat Singh, Tianyu Wang

**Abstract**: This paper begins with a description of methods for estimating image probability density functions that reflects the observation that such data is usually constrained to lie in restricted regions of the high-dimensional image space-not every pattern of pixels is an image. It is common to say that images lie on a lower-dimensional manifold in the high-dimensional space. However, it is not the case that all points on the manifold have an equal probability of being images. Images are unevenly distributed on the manifold, and our task is to devise ways to model this distribution as a probability distribution. We therefore consider popular generative models. For our purposes, generative/probabilistic models should have the properties of 1) sample generation: the possibility to sample from this distribution with the modelled density function, and 2) probability computation: given a previously unseen sample from the dataset of interest, one should be able to compute its probability, at least up to a normalising constant. To this end, we investigate the use of methods such as normalising flow and diffusion models. We then show how semantic interpretations are used to describe points on the manifold. To achieve this, we consider an emergent language framework that uses variational encoders for a disentangled representation of points that reside on a given manifold. Trajectories between points on a manifold can then be described as evolving semantic descriptions. We also show that such probabilistic descriptions (bounded) can be used to improve semantic consistency by constructing defences against adversarial attacks. We evaluate our methods with improved semantic robustness and OoD detection capability, explainable and editable semantic interpolation, and improved classification accuracy under patch attacks. We also discuss the limitation in diffusion models.



## **48. Attacks Meet Interpretability (AmI) Evaluation and Findings**

cs.CR

Need to withdraw it. The current work needs to be changed at a large  extent which would take a longer time

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.08808v3) [paper-pdf](http://arxiv.org/pdf/2310.08808v3)

**Authors**: Qian Ma, Ziping Ye, Shagufta Mehnaz

**Abstract**: To investigate the effectiveness of the model explanation in detecting adversarial examples, we reproduce the results of two papers, Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples and Is AmI (Attacks Meet Interpretability) Robust to Adversarial Examples. And then conduct experiments and case studies to identify the limitations of both works. We find that Attacks Meet Interpretability(AmI) is highly dependent on the selection of hyperparameters. Therefore, with a different hyperparameter choice, AmI is still able to detect Nicholas Carlini's attack. Finally, we propose recommendations for future work on the evaluation of defense techniques such as AmI.



## **49. HoSNN: Adversarially-Robust Homeostatic Spiking Neural Networks with Adaptive Firing Thresholds**

cs.NE

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2308.10373v2) [paper-pdf](http://arxiv.org/pdf/2308.10373v2)

**Authors**: Hejia Geng, Peng Li

**Abstract**: Spiking neural networks (SNNs) offer promise for efficient and powerful neurally inspired computation. Common to other types of neural networks, however, SNNs face the severe issue of vulnerability to adversarial attacks. We present the first study that draws inspiration from neural homeostasis to develop a bio-inspired solution that counters the susceptibilities of SNNs to adversarial onslaughts. At the heart of our approach is a novel threshold-adapting leaky integrate-and-fire (TA-LIF) neuron model, which we adopt to construct the proposed adversarially robust homeostatic SNN (HoSNN). Distinct from traditional LIF models, our TA-LIF model incorporates a self-stabilizing dynamic thresholding mechanism, curtailing adversarial noise propagation and safeguarding the robustness of HoSNNs in an unsupervised manner. Theoretical analysis is presented to shed light on the stability and convergence properties of the TA-LIF neurons, underscoring their superior dynamic robustness under input distributional shifts over traditional LIF neurons. Remarkably, without explicit adversarial training, our HoSNNs demonstrate inherent robustness on CIFAR-10, with accuracy improvements to 72.6% and 54.19% against FGSM and PGD attacks, up from 20.97% and 0.6%, respectively. Furthermore, with minimal FGSM adversarial training, our HoSNNs surpass previous models by 29.99% under FGSM and 47.83% under PGD attacks on CIFAR-10. Our findings offer a new perspective on harnessing biological principles for bolstering SNNs adversarial robustness and defense, paving the way to more resilient neuromorphic computing.



## **50. DANAA: Towards transferable attacks with double adversarial neuron attribution**

cs.CV

Accepted by 19th International Conference on Advanced Data Mining and  Applications. (ADMA 2023)

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.10427v2) [paper-pdf](http://arxiv.org/pdf/2310.10427v2)

**Authors**: Zhibo Jin, Zhiyu Zhu, Xinyi Wang, Jiayu Zhang, Jun Shen, Huaming Chen

**Abstract**: While deep neural networks have excellent results in many fields, they are susceptible to interference from attacking samples resulting in erroneous judgments. Feature-level attacks are one of the effective attack types, which targets the learnt features in the hidden layers to improve its transferability across different models. Yet it is observed that the transferability has been largely impacted by the neuron importance estimation results. In this paper, a double adversarial neuron attribution attack method, termed `DANAA', is proposed to obtain more accurate feature importance estimation. In our method, the model outputs are attributed to the middle layer based on an adversarial non-linear path. The goal is to measure the weight of individual neurons and retain the features that are more important towards transferability. We have conducted extensive experiments on the benchmark datasets to demonstrate the state-of-the-art performance of our method. Our code is available at: https://github.com/Davidjinzb/DANAA



