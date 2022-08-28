# Latest Adversarial Attack Papers
**update at 2022-08-29 06:31:21**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Semantic Preserving Adversarial Attack Generation with Autoencoder and Genetic Algorithm**

cs.LG

8 pages conference paper, accepted for publication in IEEE GLOBECOM  2022

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.12230v1)

**Authors**: Xinyi Wang, Simon Yusuf Enoch, Dong Seong Kim

**Abstracts**: Widely used deep learning models are found to have poor robustness. Little noises can fool state-of-the-art models into making incorrect predictions. While there is a great deal of high-performance attack generation methods, most of them directly add perturbations to original data and measure them using L_p norms; this can break the major structure of data, thus, creating invalid attacks. In this paper, we propose a black-box attack, which, instead of modifying original data, modifies latent features of data extracted by an autoencoder; then, we measure noises in semantic space to protect the semantics of data. We trained autoencoders on MNIST and CIFAR-10 datasets and found optimal adversarial perturbations using a genetic algorithm. Our approach achieved a 100% attack success rate on the first 100 data of MNIST and CIFAR-10 datasets with less perturbation than FGSM.



## **2. Passive Triangulation Attack on ORide**

cs.CR

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.12216v1)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstracts**: Privacy preservation in Ride Hailing Services is intended to protect privacy of drivers and riders. ORide is one of the early RHS proposals published at USENIX Security Symposium 2017. In the ORide protocol, riders and drivers, operating in a zone, encrypt their locations using a Somewhat Homomorphic Encryption scheme (SHE) and forward them to the Service Provider (SP). SP homomorphically computes the squared Euclidean distance between riders and available drivers. Rider receives the encrypted distances and selects the optimal rider after decryption. In order to prevent a triangulation attack, SP randomly permutes the distances before sending them to the rider. In this work, we use propose a passive attack that uses triangulation to determine coordinates of all participating drivers whose permuted distances are available from the points of view of multiple honest-but-curious adversary riders. An attack on ORide was published at SAC 2021. The same paper proposes a countermeasure using noisy Euclidean distances to thwart their attack. We extend our attack to determine locations of drivers when given their permuted and noisy Euclidean distances from multiple points of reference, where the noise perturbation comes from a uniform distribution. We conduct experiments with different number of drivers and for different perturbation values. Our experiments show that we can determine locations of all drivers participating in the ORide protocol. For the perturbed distance version of the ORide protocol, our algorithm reveals locations of about 25% to 50% of participating drivers. Our algorithm runs in time polynomial in number of drivers.



## **3. Automatic Mapping of Unstructured Cyber Threat Intelligence: An Experimental Study**

cs.CR

2022 IEEE 33rd International Symposium on Software Reliability  Engineering (ISSRE)

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.12144v1)

**Authors**: Vittorio Orbinato, Mariarosaria Barbaraci, Roberto Natella, Domenico Cotroneo

**Abstracts**: Proactive approaches to security, such as adversary emulation, leverage information about threat actors and their techniques (Cyber Threat Intelligence, CTI). However, most CTI still comes in unstructured forms (i.e., natural language), such as incident reports and leaked documents. To support proactive security efforts, we present an experimental study on the automatic classification of unstructured CTI into attack techniques using machine learning (ML). We contribute with two new datasets for CTI analysis, and we evaluate several ML models, including both traditional and deep learning-based ones. We present several lessons learned about how ML can perform at this task, which classifiers perform best and under which conditions, which are the main causes of classification errors, and the challenges ahead for CTI analysis.



## **4. ECG-ATK-GAN: Robustness against Adversarial Attacks on ECGs using Conditional Generative Adversarial Networks**

eess.SP

Accepted to MICCAI2022 Applications of Medical AI (AMAI) Workshop

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2110.09983v3)

**Authors**: Khondker Fariha Hossain, Sharif Amit Kamran, Alireza Tavakkoli, Xingjun Ma

**Abstracts**: Automating arrhythmia detection from ECG requires a robust and trusted system that retains high accuracy under electrical disturbances. Many machine learning approaches have reached human-level performance in classifying arrhythmia from ECGs. However, these architectures are vulnerable to adversarial attacks, which can misclassify ECG signals by decreasing the model's accuracy. Adversarial attacks are small crafted perturbations injected in the original data which manifest the out-of-distribution shifts in signal to misclassify the correct class. Thus, security concerns arise for false hospitalization and insurance fraud abusing these perturbations. To mitigate this problem, we introduce the first novel Conditional Generative Adversarial Network (GAN), robust against adversarial attacked ECG signals and retaining high accuracy. Our architecture integrates a new class-weighted objective function for adversarial perturbation identification and new blocks for discerning and combining out-of-distribution shifts in signals in the learning process for accurately classifying various arrhythmia types. Furthermore, we benchmark our architecture on six different white and black-box attacks and compare them with other recently proposed arrhythmia classification models on two publicly available ECG arrhythmia datasets. The experiment confirms that our model is more robust against such adversarial attacks for classifying arrhythmia with high accuracy.



## **5. A Perturbation Resistant Transformation and Classification System for Deep Neural Networks**

cs.CV

12 pages, 4 figures

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.11839v1)

**Authors**: Nathaniel Dean, Dilip Sarkar

**Abstracts**: Deep convolutional neural networks accurately classify a diverse range of natural images, but may be easily deceived when designed, imperceptible perturbations are embedded in the images. In this paper, we design a multi-pronged training, input transformation, and image ensemble system that is attack agnostic and not easily estimated. Our system incorporates two novel features. The first is a transformation layer that computes feature level polynomial kernels from class-level training data samples and iteratively updates input image copies at inference time based on their feature kernel differences to create an ensemble of transformed inputs. The second is a classification system that incorporates the prediction of the undefended network with a hard vote on the ensemble of filtered images. Our evaluations on the CIFAR10 dataset show our system improves the robustness of an undefended network against a variety of bounded and unbounded white-box attacks under different distance metrics, while sacrificing little accuracy on clean images. Against adaptive full-knowledge attackers creating end-to-end attacks, our system successfully augments the existing robustness of adversarially trained networks, for which our methods are most effectively applied.



## **6. A New Kind of Adversarial Example**

cs.CV

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.02430v2)

**Authors**: Ali Borji

**Abstracts**: Almost all adversarial attacks are formulated to add an imperceptible perturbation to an image in order to fool a model. Here, we consider the opposite which is adversarial examples that can fool a human but not a model. A large enough and perceptible perturbation is added to an image such that a model maintains its original decision, whereas a human will most likely make a mistake if forced to decide (or opt not to decide at all). Existing targeted attacks can be reformulated to synthesize such adversarial examples. Our proposed attack, dubbed NKE, is similar in essence to the fooling images, but is more efficient since it uses gradient descent instead of evolutionary algorithms. It also offers a new and unified perspective into the problem of adversarial vulnerability. Experimental results over MNIST and CIFAR-10 datasets show that our attack is quite efficient in fooling deep neural networks. Code is available at https://github.com/aliborji/NKE.



## **7. Attacking Neural Binary Function Detection**

cs.CR

18 pages

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11667v1)

**Authors**: Joshua Bundt, Michael Davinroy, Ioannis Agadakos, Alina Oprea, William Robertson

**Abstracts**: Binary analyses based on deep neural networks (DNNs), or neural binary analyses (NBAs), have become a hotly researched topic in recent years. DNNs have been wildly successful at pushing the performance and accuracy envelopes in the natural language and image processing domains. Thus, DNNs are highly promising for solving binary analysis problems that are typically hard due to a lack of complete information resulting from the lossy compilation process. Despite this promise, it is unclear that the prevailing strategy of repurposing embeddings and model architectures originally developed for other problem domains is sound given the adversarial contexts under which binary analysis often operates.   In this paper, we empirically demonstrate that the current state of the art in neural function boundary detection is vulnerable to both inadvertent and deliberate adversarial attacks. We proceed from the insight that current generation NBAs are built upon embeddings and model architectures intended to solve syntactic problems. We devise a simple, reproducible, and scalable black-box methodology for exploring the space of inadvertent attacks - instruction sequences that could be emitted by common compiler toolchains and configurations - that exploits this syntactic design focus. We then show that these inadvertent misclassifications can be exploited by an attacker, serving as the basis for a highly effective black-box adversarial example generation process. We evaluate this methodology against two state-of-the-art neural function boundary detectors: XDA and DeepDi. We conclude with an analysis of the evaluation data and recommendations for how future research might avoid succumbing to similar attacks.



## **8. Adversarial Driving: Attacking End-to-End Autonomous Driving**

cs.CV

7 pages, 6 figures

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2103.09151v3)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstracts**: As the research in deep neural networks advances, deep convolutional networks become feasible for automated driving tasks. There is an emerging trend of employing end-to-end models in the automation of driving tasks. However, previous research unveils that deep neural networks are vulnerable to adversarial attacks in classification tasks. While for regression tasks such as autonomous driving, the effect of these attacks remains rarely explored. In this research, we devise two white-box targeted attacks against end-to-end autonomous driving systems. The driving model takes an image as input and outputs the steering angle. Our attacks can manipulate the behavior of the autonomous driving system only by perturbing the input image. Both attacks can be initiated in real-time on CPUs without employing GPUs. This research aims to raise concerns over applications of end-to-end models in safety-critical systems.



## **9. Unrestricted Black-box Adversarial Attack Using GAN with Limited Queries**

cs.CV

Accepted to the ECCV 2022 Workshop on Adversarial Robustness in the  Real World

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11613v1)

**Authors**: Dongbin Na, Sangwoo Ji, Jong Kim

**Abstracts**: Adversarial examples are inputs intentionally generated for fooling a deep neural network. Recent studies have proposed unrestricted adversarial attacks that are not norm-constrained. However, the previous unrestricted attack methods still have limitations to fool real-world applications in a black-box setting. In this paper, we present a novel method for generating unrestricted adversarial examples using GAN where an attacker can only access the top-1 final decision of a classification model. Our method, Latent-HSJA, efficiently leverages the advantages of a decision-based attack in the latent space and successfully manipulates the latent vectors for fooling the classification model.   With extensive experiments, we demonstrate that our proposed method is efficient in evaluating the robustness of classification models with limited queries in a black-box setting. First, we demonstrate that our targeted attack method is query-efficient to produce unrestricted adversarial examples for a facial identity recognition model that contains 307 identities. Then, we demonstrate that the proposed method can also successfully attack a real-world celebrity recognition service.



## **10. Robustness of the Tangle 2.0 Consensus**

cs.DC

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.08254v2)

**Authors**: Bing-Yang Lin, Daria Dziubałtowska, Piotr Macek, Andreas Penzkofer, Sebastian Müller

**Abstracts**: In this paper, we investigate the performance of the Tangle 2.0 consensus protocol in a Byzantine environment. We use an agent-based simulation model that incorporates the main features of the Tangle 2.0 consensus protocol. Our experimental results demonstrate that the Tangle 2.0 protocol is robust to the bait-and-switch attack up to the theoretical upper bound of the adversary's 33% voting weight. We further show that the common coin mechanism in Tangle 2.0 is necessary for robustness against powerful adversaries. Moreover, the experimental results confirm that the protocol can achieve around 1s confirmation time in typical scenarios and that the confirmation times of non-conflicting transactions are not affected by the presence of conflicts.



## **11. LPF-Defense: 3D Adversarial Defense based on Frequency Analysis**

cs.CV

15 pages, 7 figures

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2202.11287v2)

**Authors**: Hanieh Naderi, Kimia Noorbakhsh, Arian Etemadi, Shohreh Kasaei

**Abstracts**: Although 3D point cloud classification has recently been widely deployed in different application scenarios, it is still very vulnerable to adversarial attacks. This increases the importance of robust training of 3D models in the face of adversarial attacks. Based on our analysis on the performance of existing adversarial attacks, more adversarial perturbations are found in the mid and high-frequency components of input data. Therefore, by suppressing the high-frequency content in the training phase, the models robustness against adversarial examples is improved. Experiments showed that the proposed defense method decreases the success rate of six attacks on PointNet, PointNet++ ,, and DGCNN models. In particular, improvements are achieved with an average increase of classification accuracy by 3.8 % on drop100 attack and 4.26 % on drop200 attack compared to the state-of-the-art methods. The method also improves models accuracy on the original dataset compared to other available methods.



## **12. Trace and Detect Adversarial Attacks on CNNs using Feature Response Maps**

cs.CV

13 pages, 6 figures

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11436v1)

**Authors**: Mohammadreza Amirian, Friedhelm Schwenker, Thilo Stadelmann

**Abstracts**: The existence of adversarial attacks on convolutional neural networks (CNN) questions the fitness of such models for serious applications. The attacks manipulate an input image such that misclassification is evoked while still looking normal to a human observer -- they are thus not easily detectable. In a different context, backpropagated activations of CNN hidden layers -- "feature responses" to a given input -- have been helpful to visualize for a human "debugger" what the CNN "looks at" while computing its output. In this work, we propose a novel detection method for adversarial examples to prevent attacks. We do so by tracking adversarial perturbations in feature responses, allowing for automatic detection using average local spatial entropy. The method does not alter the original network architecture and is fully human-interpretable. Experiments confirm the validity of our approach for state-of-the-art attacks on large-scale models trained on ImageNet.



## **13. Towards an Awareness of Time Series Anomaly Detection Models' Adversarial Vulnerability**

cs.LG

Part of Proceedings of the 31st ACM International Conference on  Information and Knowledge Management (CIKM '22)

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11264v1)

**Authors**: Shahroz Tariq, Binh M. Le, Simon S. Woo

**Abstracts**: Time series anomaly detection is extensively studied in statistics, economics, and computer science. Over the years, numerous methods have been proposed for time series anomaly detection using deep learning-based methods. Many of these methods demonstrate state-of-the-art performance on benchmark datasets, giving the false impression that these systems are robust and deployable in many practical and industrial real-world scenarios. In this paper, we demonstrate that the performance of state-of-the-art anomaly detection methods is degraded substantially by adding only small adversarial perturbations to the sensor data. We use different scoring metrics such as prediction errors, anomaly, and classification scores over several public and private datasets ranging from aerospace applications, server machines, to cyber-physical systems in power plants. Under well-known adversarial attacks from Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) methods, we demonstrate that state-of-the-art deep neural networks (DNNs) and graph neural networks (GNNs) methods, which claim to be robust against anomalies and have been possibly integrated in real-life systems, have their performance drop to as low as 0%. To the best of our understanding, we demonstrate, for the first time, the vulnerabilities of anomaly detection systems against adversarial attacks. The overarching goal of this research is to raise awareness towards the adversarial vulnerabilities of time series anomaly detectors.



## **14. ObfuNAS: A Neural Architecture Search-based DNN Obfuscation Approach**

cs.CR

9 pages

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.08569v2)

**Authors**: Tong Zhou, Shaolei Ren, Xiaolin Xu

**Abstracts**: Malicious architecture extraction has been emerging as a crucial concern for deep neural network (DNN) security. As a defense, architecture obfuscation is proposed to remap the victim DNN to a different architecture. Nonetheless, we observe that, with only extracting an obfuscated DNN architecture, the adversary can still retrain a substitute model with high performance (e.g., accuracy), rendering the obfuscation techniques ineffective. To mitigate this under-explored vulnerability, we propose ObfuNAS, which converts the DNN architecture obfuscation into a neural architecture search (NAS) problem. Using a combination of function-preserving obfuscation strategies, ObfuNAS ensures that the obfuscated DNN architecture can only achieve lower accuracy than the victim. We validate the performance of ObfuNAS with open-source architecture datasets like NAS-Bench-101 and NAS-Bench-301. The experimental results demonstrate that ObfuNAS can successfully find the optimal mask for a victim model within a given FLOPs constraint, leading up to 2.6% inference accuracy degradation for attackers with only 0.14x FLOPs overhead. The code is available at: https://github.com/Tongzhou0101/ObfuNAS.



## **15. Auditing Membership Leakages of Multi-Exit Networks**

cs.CR

Accepted by CCS 2022

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.11180v1)

**Authors**: Zheng Li, Yiyong Liu, Xinlei He, Ning Yu, Michael Backes, Yang Zhang

**Abstracts**: Relying on the fact that not all inputs require the same amount of computation to yield a confident prediction, multi-exit networks are gaining attention as a prominent approach for pushing the limits of efficient deployment. Multi-exit networks endow a backbone model with early exits, allowing to obtain predictions at intermediate layers of the model and thus save computation time and/or energy. However, current various designs of multi-exit networks are only considered to achieve the best trade-off between resource usage efficiency and prediction accuracy, the privacy risks stemming from them have never been explored. This prompts the need for a comprehensive investigation of privacy risks in multi-exit networks.   In this paper, we perform the first privacy analysis of multi-exit networks through the lens of membership leakages. In particular, we first leverage the existing attack methodologies to quantify the multi-exit networks' vulnerability to membership leakages. Our experimental results show that multi-exit networks are less vulnerable to membership leakages and the exit (number and depth) attached to the backbone model is highly correlated with the attack performance. Furthermore, we propose a hybrid attack that exploits the exit information to improve the performance of existing attacks. We evaluate membership leakage threat caused by our hybrid attack under three different adversarial setups, ultimately arriving at a model-free and data-free adversary. These results clearly demonstrate that our hybrid attacks are very broadly applicable, thereby the corresponding risks are much more severe than shown by existing membership inference attacks. We further present a defense mechanism called TimeGuard specifically for multi-exit networks and show that TimeGuard mitigates the newly proposed attacks perfectly.



## **16. Adversarial Speaker Distillation for Countermeasure Model on Automatic Speaker Verification**

cs.SD

Accepted by ISCA SPSC 2022

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2203.17031v5)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstracts**: The countermeasure (CM) model is developed to protect ASV systems from spoof attacks and prevent resulting personal information leakage in Automatic Speaker Verification (ASV) system. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud-based systems, confining the model size under a limitation. To better trade off the CM model sizes and performance, we proposed an adversarial speaker distillation method, which is an improved version of knowledge distillation method combined with generalized end-to-end (GE2E) pre-training and adversarial fine-tuning. In the evaluation phase of the ASVspoof 2021 Logical Access task, our proposed adversarial speaker distillation ResNetSE (ASD-ResNetSE) model reaches 0.2695 min t-DCF and 3.54\% EER. ASD-ResNetSE only used 22.5\% of parameters and 19.4\% of multiply and accumulate operands of ResNetSE model.



## **17. Privacy Enhancement for Cloud-Based Few-Shot Learning**

cs.LG

14 pages, 13 figures, 3 tables. Preprint. Accepted in IEEE WCCI 2022  International Joint Conference on Neural Networks (IJCNN)

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2205.07864v2)

**Authors**: Archit Parnami, Muhammad Usama, Liyue Fan, Minwoo Lee

**Abstracts**: Requiring less data for accurate models, few-shot learning has shown robustness and generality in many application domains. However, deploying few-shot models in untrusted environments may inflict privacy concerns, e.g., attacks or adversaries that may breach the privacy of user-supplied data. This paper studies the privacy enhancement for the few-shot learning in an untrusted environment, e.g., the cloud, by establishing a novel privacy-preserved embedding space that preserves the privacy of data and maintains the accuracy of the model. We examine the impact of various image privacy methods such as blurring, pixelization, Gaussian noise, and differentially private pixelization (DP-Pix) on few-shot image classification and propose a method that learns privacy-preserved representation through the joint loss. The empirical results show how privacy-performance trade-off can be negotiated for privacy-enhanced few-shot learning.



## **18. A Comprehensive Study of Real-Time Object Detection Networks Across Multiple Domains: A Survey**

cs.CV

Published in Transactions on Machine Learning Research (TMLR) with  Survey Certification

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10895v1)

**Authors**: Elahe Arani, Shruthi Gowda, Ratnajit Mukherjee, Omar Magdy, Senthilkumar Kathiresan, Bahram Zonooz

**Abstracts**: Deep neural network based object detectors are continuously evolving and are used in a multitude of applications, each having its own set of requirements. While safety-critical applications need high accuracy and reliability, low-latency tasks need resource and energy-efficient networks. Real-time detectors, which are a necessity in high-impact real-world applications, are continuously proposed, but they overemphasize the improvements in accuracy and speed while other capabilities such as versatility, robustness, resource and energy efficiency are omitted. A reference benchmark for existing networks does not exist, nor does a standard evaluation guideline for designing new networks, which results in ambiguous and inconsistent comparisons. We, thus, conduct a comprehensive study on multiple real-time detectors (anchor-, keypoint-, and transformer-based) on a wide range of datasets and report results on an extensive set of metrics. We also study the impact of variables such as image size, anchor dimensions, confidence thresholds, and architecture layers on the overall performance. We analyze the robustness of detection networks against distribution shifts, natural corruptions, and adversarial attacks. Also, we provide a calibration analysis to gauge the reliability of the predictions. Finally, to highlight the real-world impact, we conduct two unique case studies, on autonomous driving and healthcare applications. To further gauge the capability of networks in critical real-time applications, we report the performance after deploying the detection networks on edge devices. Our extensive empirical study can act as a guideline for the industrial community to make an informed choice on the existing networks. We also hope to inspire the research community towards a new direction in the design and evaluation of networks that focuses on a bigger and holistic overview for a far-reaching impact.



## **19. Transferability Ranking of Adversarial Examples**

cs.LG

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10878v1)

**Authors**: Mosh Levy, Yuval Elovici, Yisroel Mirsky

**Abstracts**: Adversarial examples can be used to maliciously and covertly change a model's prediction. It is known that an adversarial example designed for one model can transfer to other models as well. This poses a major threat because it means that attackers can target systems in a blackbox manner.   In the domain of transferability, researchers have proposed ways to make attacks more transferable and to make models more robust to transferred examples. However, to the best of our knowledge, there are no works which propose a means for ranking the transferability of an adversarial example in the perspective of a blackbox attacker. This is an important task because an attacker is likely to use only a select set of examples, and therefore will want to select the samples which are most likely to transfer.   In this paper we suggest a method for ranking the transferability of adversarial examples without access to the victim's model. To accomplish this, we define and estimate the expected transferability of a sample given limited information about the victim. We also explore practical scenarios: where the adversary can select the best sample to attack and where the adversary must use a specific sample but can choose different perturbations. Through our experiments, we found that our ranking method can increase an attacker's success rate by up to 80% compared to the baseline (random selection without ranking).



## **20. Complete Traceability Multimedia Fingerprinting Codes Resistant to Averaging Attack and Adversarial Noise with Optimal Rate**

cs.IT

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2108.09015v4)

**Authors**: Ilya Vorobyev

**Abstracts**: In this paper we consider complete traceability multimedia fingerprinting codes resistant to averaging attacks and adversarial noise. Recently it was shown that there are no such codes for the case of an arbitrary linear attack. However, for the case of averaging attacks complete traceability multimedia fingerprinting codes of exponential cardinality resistant to constant adversarial noise were constructed in 2020 by Egorova et al. We continue this work and provide an improved lower bound on the rate of these codes.



## **21. Evaluating Machine Unlearning via Epistemic Uncertainty**

cs.LG

Rejected at ECML 2021. Even though the paper was rejected, we want to  "publish" it on arxiv, since we believe that it is nevertheless interesting  to investigate the connections between unlearning and uncertainty

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10836v1)

**Authors**: Alexander Becker, Thomas Liebig

**Abstracts**: There has been a growing interest in Machine Unlearning recently, primarily due to legal requirements such as the General Data Protection Regulation (GDPR) and the California Consumer Privacy Act. Thus, multiple approaches were presented to remove the influence of specific target data points from a trained model. However, when evaluating the success of unlearning, current approaches either use adversarial attacks or compare their results to the optimal solution, which usually incorporates retraining from scratch. We argue that both ways are insufficient in practice. In this work, we present an evaluation metric for Machine Unlearning algorithms based on epistemic uncertainty. This is the first definition of a general evaluation metric for Machine Unlearning to our best knowledge.



## **22. UKP-SQuARE v2 Explainability and Adversarial Attacks for Trustworthy QA**

cs.CL

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.09316v2)

**Authors**: Rachneet Sachdeva, Haritz Puerto, Tim Baumgärtner, Sewin Tariverdian, Hao Zhang, Kexin Wang, Hossain Shaikh Saadi, Leonardo F. R. Ribeiro, Iryna Gurevych

**Abstracts**: Question Answering (QA) systems are increasingly deployed in applications where they support real-world decisions. However, state-of-the-art models rely on deep neural networks, which are difficult to interpret by humans. Inherently interpretable models or post hoc explainability methods can help users to comprehend how a model arrives at its prediction and, if successful, increase their trust in the system. Furthermore, researchers can leverage these insights to develop new methods that are more accurate and less biased. In this paper, we introduce SQuARE v2, the new version of SQuARE, to provide an explainability infrastructure for comparing models based on methods such as saliency maps and graph-based explanations. While saliency maps are useful to inspect the importance of each input token for the model's prediction, graph-based explanations from external Knowledge Graphs enable the users to verify the reasoning behind the model prediction. In addition, we provide multiple adversarial attacks to compare the robustness of QA models. With these explainability methods and adversarial attacks, we aim to ease the research on trustworthy QA models. SQuARE is available on https://square.ukp-lab.de.



## **23. SoK: Certified Robustness for Deep Neural Networks**

cs.LG

To appear at 2023 IEEE Symposium on Security and Privacy (SP); 14  pages for the main text; benchmark & tool website:  http://sokcertifiedrobustness.github.io/

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2009.04131v7)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which can usually be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches, which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate 20+ representative certifiably robust approaches.



## **24. Adversarial Vulnerability of Temporal Feature Networks for Object Detection**

cs.CV

Accepted for publication at ECCV 2022 SAIAD workshop

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10773v1)

**Authors**: Svetlana Pavlitskaya, Nikolai Polley, Michael Weber, J. Marius Zöllner

**Abstracts**: Taking into account information across the temporal domain helps to improve environment perception in autonomous driving. However, it has not been studied so far whether temporally fused neural networks are vulnerable to deliberately generated perturbations, i.e. adversarial attacks, or whether temporal history is an inherent defense against them. In this work, we study whether temporal feature networks for object detection are vulnerable to universal adversarial attacks. We evaluate attacks of two types: imperceptible noise for the whole image and locally-bound adversarial patch. In both cases, perturbations are generated in a white-box manner using PGD. Our experiments confirm, that attacking even a portion of a temporal input suffices to fool the network. We visually assess generated perturbations to gain insights into the functioning of attacks. To enhance the robustness, we apply adversarial training using 5-PGD. Our experiments on KITTI and nuScenes datasets demonstrate, that a model robustified via K-PGD is able to withstand the studied attacks while keeping the mAP-based performance comparable to that of an unattacked model.



## **25. MALICE: Manipulation Attacks on Learned Image ComprEssion**

cs.CV

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2205.13253v2)

**Authors**: Kang Liu, Di Wu, Yiru Wang, Dan Feng, Benjamin Tan, Siddharth Garg

**Abstracts**: Deep learning techniques have shown promising results in image compression, with competitive bitrate and image reconstruction quality from compressed latent. However, while image compression has progressed towards a higher peak signal-to-noise ratio (PSNR) and fewer bits per pixel (bpp), their robustness to adversarial images has never received deliberation. In this work, we, for the first time, investigate the robustness of image compression systems where imperceptible perturbation of input images can precipitate a significant increase in the bitrate of their compressed latent. To characterize the robustness of state-of-the-art learned image compression, we mount white-box and black-box attacks. Our white-box attack employs fast gradient sign method on the entropy estimation of the bitstream as its bitrate approximation. We propose DCT-Net simulating JPEG compression with architectural simplicity and lightweight training as the substitute in the black-box attack and enable fast adversarial transferability. Our results on six image compression models, each with six different bitrate qualities (thirty-six models in total), show that they are surprisingly fragile, where the white-box attack achieves up to 56.326x and black-box 1.947x bpp change. To improve robustness, we propose a novel compression architecture factorAtn which incorporates attention modules and a basic factorized entropy model, resulting in a promising trade-off between the rate-distortion performance and robustness to adversarial attacks that surpasses existing learned image compressors.



## **26. RAB: Provable Robustness Against Backdoor Attacks**

cs.LG

IEEE Symposium on Security and Privacy 2023

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2003.08904v7)

**Authors**: Maurice Weber, Xiaojun Xu, Bojan Karlaš, Ce Zhang, Bo Li

**Abstracts**: Recent studies have shown that deep neural networks (DNNs) are vulnerable to adversarial attacks, including evasion and backdoor (poisoning) attacks. On the defense side, there have been intensive efforts on improving both empirical and provable robustness against evasion attacks; however, the provable robustness against backdoor attacks still remains largely unexplored. In this paper, we focus on certifying the machine learning model robustness against general threat models, especially backdoor attacks. We first provide a unified framework via randomized smoothing techniques and show how it can be instantiated to certify the robustness against both evasion and backdoor attacks. We then propose the first robust training process, RAB, to smooth the trained model and certify its robustness against backdoor attacks. We prove the robustness bound for machine learning models trained with RAB and prove that our robustness bound is tight. In addition, we theoretically show that it is possible to train the robust smoothed models efficiently for simple models such as K-nearest neighbor classifiers, and we propose an exact smooth-training algorithm that eliminates the need to sample from a noise distribution for such models. Empirically, we conduct comprehensive experiments for different machine learning (ML) models such as DNNs, support vector machines, and K-NN models on MNIST, CIFAR-10, and ImageNette datasets and provide the first benchmark for certified robustness against backdoor attacks. In addition, we evaluate K-NN models on a spambase tabular dataset to demonstrate the advantages of the proposed exact algorithm. Both the theoretic analysis and the comprehensive evaluation on diverse ML models and datasets shed light on further robust learning strategies against general training time attacks.



## **27. Hierarchical Perceptual Noise Injection for Social Media Fingerprint Privacy Protection**

cs.CV

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10688v1)

**Authors**: Simin Li, Huangxinxin Xu, Jiakai Wang, Aishan Liu, Fazhi He, Xianglong Liu, Dacheng Tao

**Abstracts**: Billions of people are sharing their daily life images on social media every day. However, their biometric information (e.g., fingerprint) could be easily stolen from these images. The threat of fingerprint leakage from social media raises a strong desire for anonymizing shared images while maintaining image qualities, since fingerprints act as a lifelong individual biometric password. To guard the fingerprint leakage, adversarial attack emerges as a solution by adding imperceptible perturbations on images. However, existing works are either weak in black-box transferability or appear unnatural. Motivated by visual perception hierarchy (i.e., high-level perception exploits model-shared semantics that transfer well across models while low-level perception extracts primitive stimulus and will cause high visual sensitivities given suspicious stimulus), we propose FingerSafe, a hierarchical perceptual protective noise injection framework to address the mentioned problems. For black-box transferability, we inject protective noises on fingerprint orientation field to perturb the model-shared high-level semantics (i.e., fingerprint ridges). Considering visual naturalness, we suppress the low-level local contrast stimulus by regularizing the response of Lateral Geniculate Nucleus. Our FingerSafe is the first to provide feasible fingerprint protection in both digital (up to 94.12%) and realistic scenarios (Twitter and Facebook, up to 68.75%). Our code can be found at https://github.com/nlsde-safety-team/FingerSafe.



## **28. Efficient Detection and Filtering Systems for Distributed Training**

cs.LG

18 pages, 14 figures, 6 tables. The material in this work appeared in  part at arXiv:2108.02416 which has been published at the 2022 IEEE  International Symposium on Information Theory

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.08085v3)

**Authors**: Konstantinos Konstantinidis, Aditya Ramamoorthy

**Abstracts**: A plethora of modern machine learning tasks requires the utilization of large-scale distributed clusters as a critical component of the training pipeline. However, abnormal Byzantine behavior of the worker nodes can derail the training and compromise the quality of the inference. Such behavior can be attributed to unintentional system malfunctions or orchestrated attacks; as a result, some nodes may return arbitrary results to the parameter server (PS) that coordinates the training. Recent work considers a wide range of attack models and has explored robust aggregation and/or computational redundancy to correct the distorted gradients.   In this work, we consider attack models ranging from strong ones: $q$ omniscient adversaries with full knowledge of the defense protocol that can change from iteration to iteration to weak ones: $q$ randomly chosen adversaries with limited collusion abilities that only change every few iterations at a time. Our algorithms rely on redundant task assignments coupled with detection of adversarial behavior. For strong attacks, we demonstrate a reduction in the fraction of distorted gradients ranging from 16%-99% as compared to the prior state-of-the-art. Our top-1 classification accuracy results on the CIFAR-10 data set demonstrate a 25% advantage in accuracy (averaged over strong and weak scenarios) under the most sophisticated attacks compared to state-of-the-art methods.



## **29. Optimal Bootstrapping of PoW Blockchains**

cs.CR

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.10618v1)

**Authors**: Ranvir Rana, Dimitris Karakostas, Sreeram Kannan, Aggelos Kiayias, Pramod Viswanath

**Abstracts**: Proof of Work (PoW) blockchains are susceptible to adversarial majority mining attacks in the early stages due to incipient participation and corresponding low net hash power. Bootstrapping ensures safety and liveness during the transient stage by protecting against a majority mining attack, allowing a PoW chain to grow the participation base and corresponding mining hash power. Liveness is especially important since a loss of liveness will lead to loss of honest mining rewards, decreasing honest participation, hence creating an undesired spiral; indeed existing bootstrapping mechanisms offer especially weak liveness guarantees.   In this paper, we propose Advocate, a new bootstrapping methodology, which achieves two main results: (a) optimal liveness and low latency under a super-majority adversary for the Nakamoto longest chain protocol and (b) immediate black-box generalization to a variety of parallel-chain based scaling architectures, including OHIE and Prism. We demonstrate via a full-stack implementation the robustness of Advocate under a 90% adversarial majority.



## **30. Different Spectral Representations in Optimized Artificial Neural Networks and Brains**

cs.LG

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.10576v1)

**Authors**: Richard C. Gerum, Cassidy Pirlot, Alona Fyshe, Joel Zylberberg

**Abstracts**: Recent studies suggest that artificial neural networks (ANNs) that match the spectral properties of the mammalian visual cortex -- namely, the $\sim 1/n$ eigenspectrum of the covariance matrix of neural activities -- achieve higher object recognition performance and robustness to adversarial attacks than those that do not. To our knowledge, however, no previous work systematically explored how modifying the ANN's spectral properties affects performance. To fill this gap, we performed a systematic search over spectral regularizers, forcing the ANN's eigenspectrum to follow $1/n^\alpha$ power laws with different exponents $\alpha$. We found that larger powers (around 2--3) lead to better validation accuracy and more robustness to adversarial attacks on dense networks. This surprising finding applied to both shallow and deep networks and it overturns the notion that the brain-like spectrum (corresponding to $\alpha \sim 1$) always optimizes ANN performance and/or robustness. For convolutional networks, the best $\alpha$ values depend on the task complexity and evaluation metric: lower $\alpha$ values optimized validation accuracy and robustness to adversarial attack for networks performing a simple object recognition task (categorizing MNIST images of handwritten digits); for a more complex task (categorizing CIFAR-10 natural images), we found that lower $\alpha$ values optimized validation accuracy whereas higher $\alpha$ values optimized adversarial robustness. These results have two main implications. First, they cast doubt on the notion that brain-like spectral properties ($\alpha \sim 1$) \emph{always} optimize ANN performance. Second, they demonstrate the potential for fine-tuned spectral regularizers to optimize a chosen design metric, i.e., accuracy and/or robustness.



## **31. On the Decision Boundaries of Neural Networks: A Tropical Geometry Perspective**

cs.LG

First two authors contributed equally to this work

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2002.08838v3)

**Authors**: Motasem Alfarra, Adel Bibi, Hasan Hammoud, Mohamed Gaafar, Bernard Ghanem

**Abstracts**: This work tackles the problem of characterizing and understanding the decision boundaries of neural networks with piecewise linear non-linearity activations. We use tropical geometry, a new development in the area of algebraic geometry, to characterize the decision boundaries of a simple network of the form (Affine, ReLU, Affine). Our main finding is that the decision boundaries are a subset of a tropical hypersurface, which is intimately related to a polytope formed by the convex hull of two zonotopes. The generators of these zonotopes are functions of the network parameters. This geometric characterization provides new perspectives to three tasks. (i) We propose a new tropical perspective to the lottery ticket hypothesis, where we view the effect of different initializations on the tropical geometric representation of a network's decision boundaries. (ii) Moreover, we propose new tropical based optimization reformulations that directly influence the decision boundaries of the network for the task of network pruning. (iii) At last, we discuss the reformulation of the generation of adversarial attacks in a tropical sense. We demonstrate that one can construct adversaries in a new tropical setting by perturbing a specific set of decision boundaries by perturbing a set of parameters in the network.



## **32. Toward Better Target Representation for Source-Free and Black-Box Domain Adaptation**

cs.CV

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.10531v1)

**Authors**: Qucheng Peng, Zhengming Ding, Lingjuan Lyu, Lichao Sun, Chen Chen

**Abstracts**: Domain adaptation aims at aligning the labeled source domain and the unlabeled target domain, and most existing approaches assume the source data is accessible. Unfortunately, this paradigm raises concerns in data privacy and security. Recent studies try to dispel these concerns by the Source-Free setting, which adapts the source-trained model towards target domain without exposing the source data. However, the Source-Free paradigm is still at risk of data leakage due to adversarial attacks to the source model. Hence, the Black-Box setting is proposed, where only the outputs of source model can be utilized. In this paper, we address both the Source-Free adaptation and the Black-Box adaptation, proposing a novel method named better target representation from Frequency Mixup and Mutual Learning (FMML). Specifically, we introduce a new data augmentation technique as Frequency MixUp, which highlights task-relevant objects in the interpolations, thus enhancing class-consistency and linear behavior for target models. Moreover, we introduce a network regularization method called Mutual Learning to the domain adaptation problem. It transfers knowledge inside the target model via self-knowledge distillation and thus alleviates overfitting on the source domain by learning multi-scale target representations. Extensive experiments show that our method achieves state-of-the-art performance on several benchmark datasets under both settings.



## **33. BARReL: Bottleneck Attention for Adversarial Robustness in Vision-Based Reinforcement Learning**

cs.LG

5 pages, 2 figures, 3 tables

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.10481v1)

**Authors**: Eugene Bykovets, Yannick Metz, Mennatallah El-Assady, Daniel A. Keim, Joachim M. Buhmann

**Abstracts**: Robustness to adversarial perturbations has been explored in many areas of computer vision. This robustness is particularly relevant in vision-based reinforcement learning, as the actions of autonomous agents might be safety-critic or impactful in the real world. We investigate the susceptibility of vision-based reinforcement learning agents to gradient-based adversarial attacks and evaluate a potential defense. We observe that Bottleneck Attention Modules (BAM) included in CNN architectures can act as potential tools to increase robustness against adversarial attacks. We show how learned attention maps can be used to recover activations of a convolutional layer by restricting the spatial activations to salient regions. Across a number of RL environments, BAM-enhanced architectures show increased robustness during inference. Finally, we discuss potential future research directions.



## **34. Membership-Doctor: Comprehensive Assessment of Membership Inference Against Machine Learning Models**

cs.CR

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.10445v1)

**Authors**: Xinlei He, Zheng Li, Weilin Xu, Cory Cornelius, Yang Zhang

**Abstracts**: Machine learning models are prone to memorizing sensitive data, making them vulnerable to membership inference attacks in which an adversary aims to infer whether an input sample was used to train the model. Over the past few years, researchers have produced many membership inference attacks and defenses. However, these attacks and defenses employ a variety of strategies and are conducted in different models and datasets. The lack of comprehensive benchmark, however, means we do not understand the strengths and weaknesses of existing attacks and defenses.   We fill this gap by presenting a large-scale measurement of different membership inference attacks and defenses. We systematize membership inference through the study of nine attacks and six defenses and measure the performance of different attacks and defenses in the holistic evaluation. We then quantify the impact of the threat model on the results of these attacks. We find that some assumptions of the threat model, such as same-architecture and same-distribution between shadow and target models, are unnecessary. We are also the first to execute attacks on the real-world data collected from the Internet, instead of laboratory datasets. We further investigate what determines the performance of membership inference attacks and reveal that the commonly believed overfitting level is not sufficient for the success of the attacks. Instead, the Jensen-Shannon distance of entropy/cross-entropy between member and non-member samples correlates with attack performance much better. This gives us a new way to accurately predict membership inference risks without running the attack. Finally, we find that data augmentation degrades the performance of existing attacks to a larger extent, and we propose an adaptive attack using augmentation to train shadow and attack models that improve attack performance.



## **35. On Deep Learning in Password Guessing, a Survey**

cs.CR

8 pages, 4 figures, 3 tables. arXiv admin note: substantial text  overlap with arXiv:2208.06943

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.10413v1)

**Authors**: Fangyi Yu

**Abstracts**: The security of passwords is dependent on a thorough understanding of the strategies used by attackers. Unfortunately, real-world adversaries use pragmatic guessing tactics like dictionary attacks, which are difficult to simulate in password security research. Dictionary attacks must be carefully configured and modified to be representative of the actual threat. This approach, however, needs domain-specific knowledge and expertise that are difficult to duplicate. This paper compares various deep learning-based password guessing approaches that do not require domain knowledge or assumptions about users' password structures and combinations. The involved model categories are Recurrent Neural Networks, Generative Adversarial Networks, Autoencoder, and Attention mechanisms. Additionally, we proposed a promising research experimental design on using variations of IWGAN on password guessing under non-targeted offline attacks. Using these advanced strategies, we can enhance password security and create more accurate and efficient Password Strength Meters.



## **36. Fight Fire With Fire: Reversing Skin Adversarial Examples by Multiscale Diffusive and Denoising Aggregation Mechanism**

cs.CV

11 pages

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.10373v1)

**Authors**: Yongwei Wang, Yuan Li, Zhiqi Shen

**Abstracts**: Reliable skin cancer diagnosis models play an essential role in early screening and medical intervention. Prevailing computer-aided skin cancer classification systems employ deep learning approaches. However, recent studies reveal their extreme vulnerability to adversarial attacks -- often imperceptible perturbations to significantly reduce performances of skin cancer diagnosis models. To mitigate these threats, this work presents a simple, effective and resource-efficient defense framework by reverse engineering adversarial perturbations in skin cancer images. Specifically, a multiscale image pyramid is first established to better preserve discriminative structures in medical imaging domain. To neutralize adversarial effects, skin images at different scales are then progressively diffused by injecting isotropic Gaussian noises to move the adversarial examples to the clean image manifold. Crucially, to further reverse adversarial noises and suppress redundant injected noises, a novel multiscale denoising mechanism is carefully designed that aggregates image information from neighboring scales. We evaluated the defensive effectiveness of our method on ISIC 2019, a largest skin cancer multiclass classification dataset. Experimental results demonstrate that the proposed method can successfully reverse adversarial perturbations from different attacks and significantly outperform some state-of-the-art methods in defending skin cancer diagnosis models.



## **37. Adversarial Classification under Gaussian Mechanism: Calibrating the Attack to Sensitivity**

cs.IT

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2201.09751v4)

**Authors**: Ayse Unsal, Melek Onen

**Abstracts**: This work studies anomaly detection under differential privacy (DP) with Gaussian perturbation using both statistical and information-theoretic tools. In our setting, the adversary aims to modify the content of a statistical dataset by inserting additional data without being detected by using the DP guarantee to her own benefit. To this end, we characterize information-theoretic and statistical thresholds for the first and second-order statistics of the adversary's attack, which balances the privacy budget and the impact of the attack in order to remain undetected. Additionally, we introduce a new privacy metric based on Chernoff information for classifying adversaries under differential privacy as a stronger alternative to $(\epsilon, \delta)-$ and Kullback-Leibler DP for the Gaussian mechanism. Analytical results are supported by numerical evaluations.



## **38. On Adversarial Robustness of Trajectory Prediction for Autonomous Vehicles**

cs.CV

13 pages, 13 figures, accepted by CVPR 2022

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2201.05057v3)

**Authors**: Qingzhao Zhang, Shengtuo Hu, Jiachen Sun, Qi Alfred Chen, Z. Morley Mao

**Abstracts**: Trajectory prediction is a critical component for autonomous vehicles (AVs) to perform safe planning and navigation. However, few studies have analyzed the adversarial robustness of trajectory prediction or investigated whether the worst-case prediction can still lead to safe planning. To bridge this gap, we study the adversarial robustness of trajectory prediction models by proposing a new adversarial attack that perturbs normal vehicle trajectories to maximize the prediction error. Our experiments on three models and three datasets show that the adversarial prediction increases the prediction error by more than 150%. Our case studies show that if an adversary drives a vehicle close to the target AV following the adversarial trajectory, the AV may make an inaccurate prediction and even make unsafe driving decisions. We also explore possible mitigation techniques via data augmentation and trajectory smoothing. The implementation is open source at https://github.com/zqzqz/AdvTrajectoryPrediction.



## **39. Inferring Sensitive Attributes from Model Explanations**

cs.CR

ACM CIKM 2022

**SubmitDate**: 2022-08-21    [paper-pdf](http://arxiv.org/pdf/2208.09967v1)

**Authors**: Vasisht Duddu, Antoine Boutet

**Abstracts**: Model explanations provide transparency into a trained machine learning model's blackbox behavior to a model builder. They indicate the influence of different input attributes to its corresponding model prediction. The dependency of explanations on input raises privacy concerns for sensitive user data. However, current literature has limited discussion on privacy risks of model explanations.   We focus on the specific privacy risk of attribute inference attack wherein an adversary infers sensitive attributes of an input (e.g., race and sex) given its model explanations. We design the first attribute inference attack against model explanations in two threat models where model builder either (a) includes the sensitive attributes in training data and input or (b) censors the sensitive attributes by not including them in the training data and input.   We evaluate our proposed attack on four benchmark datasets and four state-of-the-art algorithms. We show that an adversary can successfully infer the value of sensitive attributes from explanations in both the threat models accurately. Moreover, the attack is successful even by exploiting only the explanations corresponding to sensitive attributes. These suggest that our attack is effective against explanations and poses a practical threat to data privacy.   On combining the model predictions (an attack surface exploited by prior attacks) with explanations, we note that the attack success does not improve. Additionally, the attack success on exploiting model explanations is better compared to exploiting only model predictions. These suggest that model explanations are a strong attack surface to exploit for an adversary.



## **40. Increasing-Margin Adversarial (IMA) Training to Improve Adversarial Robustness of Neural Networks**

cs.CV

12 pages

**SubmitDate**: 2022-08-21    [paper-pdf](http://arxiv.org/pdf/2005.09147v9)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: Deep neural networks (DNNs) are vulnerable to adversarial noises. By adding adversarial noises to training samples, adversarial training can improve the model's robustness against adversarial noises. However, adversarial training samples with excessive noises can harm standard accuracy, which may be unacceptable for many medical image analysis applications. This issue has been termed the trade-off between standard accuracy and adversarial robustness. In this paper, we hypothesize that this issue may be alleviated if the adversarial samples for training are placed right on the decision boundaries. Based on this hypothesis, we design an adaptive adversarial training method, named IMA. For each individual training sample, IMA makes a sample-wise estimation of the upper bound of the adversarial perturbation. In the training process, each of the sample-wise adversarial perturbations is gradually increased to match the margin. Once an equilibrium state is reached, the adversarial perturbations will stop increasing. IMA is evaluated on publicly available datasets under two popular adversarial attacks, PGD and IFGSM. The results show that: (1) IMA significantly improves adversarial robustness of DNN classifiers, which achieves state-of-the-art performance; (2) IMA has a minimal reduction in clean accuracy among all competing defense methods; (3) IMA can be applied to pretrained models to reduce time cost; (4) IMA can be applied to the state-of-the-art medical image segmentation networks, with outstanding performance. We hope our work may help to lift the trade-off between adversarial robustness and clean accuracy and facilitate the development of robust applications in the medical field. The source code will be released when this paper is published.



## **41. MockingBERT: A Method for Retroactively Adding Resilience to NLP Models**

cs.CL

8 pages (excl. bibiography and appendix), 2 figures The code  necessary for reproduction is available at  https://github.com/akash13singh/resilient_nlp To be published in Proceedings  of the 29th International Conference on Computational Linguistics (COLING  2022)

**SubmitDate**: 2022-08-21    [paper-pdf](http://arxiv.org/pdf/2208.09915v1)

**Authors**: Jan Jezabek, Akash Singh

**Abstracts**: Protecting NLP models against misspellings whether accidental or adversarial has been the object of research interest for the past few years. Existing remediations have typically either compromised accuracy or required full model re-training with each new class of attacks. We propose a novel method of retroactively adding resilience to misspellings to transformer-based NLP models. This robustness can be achieved without the need for re-training of the original NLP model and with only a minimal loss of language understanding performance on inputs without misspellings. Additionally we propose a new efficient approximate method of generating adversarial misspellings, which significantly reduces the cost needed to evaluate a model's resilience to adversarial attacks.



## **42. On The Robustness of Channel Allocation in Joint Radar And Communication Systems: An Auction Approach**

cs.GT

**SubmitDate**: 2022-08-21    [paper-pdf](http://arxiv.org/pdf/2208.09821v1)

**Authors**: Ismail Lotfi, Hongyang Du, Dusit Niyato, Sumei Sun, Dong In Kim

**Abstracts**: Joint radar and communication (JRC) is a promising technique for spectrum re-utilization, which enables radar sensing and data transmission to operate on the same frequencies and the same devices. However, due to the multi-objective property of JRC systems, channel allocation to JRC nodes should be carefully designed to maximize system performance. Additionally, because of the broadcast nature of wireless signals, a watchful adversary, i.e., a warden, can detect ongoing transmissions and attack the system. Thus, we develop a covert JRC system that minimizes the detection probability by wardens, in which friendly jammers are deployed to improve the covertness of the JRC nodes during radar sensing and data transmission operations. Furthermore, we propose a robust multi-item auction design for channel allocation for such a JRC system that considers the uncertainty in bids. The proposed auction mechanism achieves the properties of truthfulness, individual rationality, budget feasibility, and computational efficiency. The simulations clearly show the benefits of our design to support covert JRC systems and to provide incentive to the JRC nodes in obtaining spectrum, in which the auction-based channel allocation mechanism is robust against perturbations in the bids, which is highly effective for JRC nodes working in uncertain environments.



## **43. PointDP: Diffusion-driven Purification against Adversarial Attacks on 3D Point Cloud Recognition**

cs.CV

**SubmitDate**: 2022-08-21    [paper-pdf](http://arxiv.org/pdf/2208.09801v1)

**Authors**: Jiachen Sun, Weili Nie, Zhiding Yu, Z. Morley Mao, Chaowei Xiao

**Abstracts**: 3D Point cloud is becoming a critical data representation in many real-world applications like autonomous driving, robotics, and medical imaging. Although the success of deep learning further accelerates the adoption of 3D point clouds in the physical world, deep learning is notorious for its vulnerability to adversarial attacks. In this work, we first identify that the state-of-the-art empirical defense, adversarial training, has a major limitation in applying to 3D point cloud models due to gradient obfuscation. We further propose PointDP, a purification strategy that leverages diffusion models to defend against 3D adversarial attacks. We extensively evaluate PointDP on six representative 3D point cloud architectures, and leverage 10+ strong and adaptive attacks to demonstrate its lower-bound robustness. Our evaluation shows that PointDP achieves significantly better robustness than state-of-the-art purification methods under strong attacks. Results of certified defenses on randomized smoothing combined with PointDP will be included in the near future.



## **44. Robust Node Classification on Graphs: Jointly from Bayesian Label Transition and Topology-based Label Propagation**

cs.LG

The paper is accepted for CIKM 2022

**SubmitDate**: 2022-08-21    [paper-pdf](http://arxiv.org/pdf/2208.09779v1)

**Authors**: Jun Zhuang, Mohammad Al Hasan

**Abstracts**: Node classification using Graph Neural Networks (GNNs) has been widely applied in various real-world scenarios. However, in recent years, compelling evidence emerges that the performance of GNN-based node classification may deteriorate substantially by topological perturbation, such as random connections or adversarial attacks. Various solutions, such as topological denoising methods and mechanism design methods, have been proposed to develop robust GNN-based node classifiers but none of these works can fully address the problems related to topological perturbations. Recently, the Bayesian label transition model is proposed to tackle this issue but its slow convergence may lead to inferior performance. In this work, we propose a new label inference model, namely LInDT, which integrates both Bayesian label transition and topology-based label propagation for improving the robustness of GNNs against topological perturbations. LInDT is superior to existing label transition methods as it improves the label prediction of uncertain nodes by utilizing neighborhood-based label propagation leading to better convergence of label inference. Besides, LIndT adopts asymmetric Dirichlet distribution as a prior, which also helps it to improve label inference. Extensive experiments on five graph datasets demonstrate the superiority of LInDT for GNN-based node classification under three scenarios of topological perturbations.



## **45. GAIROSCOPE: Injecting Data from Air-Gapped Computers to Nearby Gyroscopes**

cs.CR

**SubmitDate**: 2022-08-21    [paper-pdf](http://arxiv.org/pdf/2208.09764v1)

**Authors**: Mordechai Guri

**Abstracts**: It is known that malware can leak data from isolated, air-gapped computers to nearby smartphones using ultrasonic waves. However, this covert channel requires access to the smartphone's microphone, which is highly protected in Android OS and iOS, and might be non-accessible, disabled, or blocked.   In this paper we present `GAIROSCOPE,' an ultrasonic covert channel that doesn't require a microphone on the receiving side. Our malware generates ultrasonic tones in the resonance frequencies of the MEMS gyroscope. These inaudible frequencies produce tiny mechanical oscillations within the smartphone's gyroscope, which can be demodulated into binary information. Notably, the gyroscope in smartphones is considered to be a 'safe' sensor that can be used legitimately from mobile apps and javascript. We introduce the adversarial attack model and present related work. We provide the relevant technical background and show the design and implementation of GAIROSCOPE. We present the evaluation results and discuss a set of countermeasures to this threat. Our experiments show that attackers can exfiltrate sensitive information from air-gapped computers to smartphones located a few meters away via Speakers-to-Gyroscope covert channel.



## **46. GAT: Generative Adversarial Training for Adversarial Example Detection and Robust Classification**

cs.LG

ICLR 2020, code is available at  https://github.com/xuwangyin/GAT-Generative-Adversarial-Training

**SubmitDate**: 2022-08-20    [paper-pdf](http://arxiv.org/pdf/1905.11475v3)

**Authors**: Xuwang Yin, Soheil Kolouri, Gustavo K. Rohde

**Abstracts**: The vulnerabilities of deep neural networks against adversarial examples have become a significant concern for deploying these models in sensitive domains. Devising a definitive defense against such attacks is proven to be challenging, and the methods relying on detecting adversarial samples are only valid when the attacker is oblivious to the detection mechanism. In this paper we propose a principled adversarial example detection method that can withstand norm-constrained white-box attacks. Inspired by one-versus-the-rest classification, in a K class classification problem, we train K binary classifiers where the i-th binary classifier is used to distinguish between clean data of class i and adversarially perturbed samples of other classes. At test time, we first use a trained classifier to get the predicted label (say k) of the input, and then use the k-th binary classifier to determine whether the input is a clean sample (of class k) or an adversarially perturbed example (of other classes). We further devise a generative approach to detecting/classifying adversarial examples by interpreting each binary classifier as an unnormalized density model of the class-conditional data. We provide comprehensive evaluation of the above adversarial example detection/classification methods, and demonstrate their competitive performances and compelling properties.



## **47. Analyzing Adversarial Robustness of Vision Transformers against Spatial and Spectral Attacks**

cs.CV

11 pages, 13 figures

**SubmitDate**: 2022-08-20    [paper-pdf](http://arxiv.org/pdf/2208.09602v1)

**Authors**: Gihyun Kim, Jong-Seok Lee

**Abstracts**: Vision Transformers have emerged as a powerful architecture that can outperform convolutional neural networks (CNNs) in image classification tasks. Several attempts have been made to understand robustness of Transformers against adversarial attacks, but existing studies draw inconsistent results, i.e., some conclude that Transformers are more robust than CNNs, while some others find that they have similar degrees of robustness. In this paper, we address two issues unexplored in the existing studies examining adversarial robustness of Transformers. First, we argue that the image quality should be simultaneously considered in evaluating adversarial robustness. We find that the superiority of one architecture to another in terms of robustness can change depending on the attack strength expressed by the quality of the attacked images. Second, by noting that Transformers and CNNs rely on different types of information in images, we formulate an attack framework, called Fourier attack, as a tool for implementing flexible attacks, where an image can be attacked in the spectral domain as well as in the spatial domain. This attack perturbs the magnitude and phase information of particular frequency components selectively. Through extensive experiments, we find that Transformers tend to rely more on phase information and low frequency information than CNNs, and thus sometimes they are even more vulnerable under frequency-selective attacks. It is our hope that this work provides new perspectives in understanding the properties and adversarial robustness of Transformers.



## **48. Gender Bias and Universal Substitution Adversarial Attacks on Grammatical Error Correction Systems for Automated Assessment**

cs.CL

**SubmitDate**: 2022-08-19    [paper-pdf](http://arxiv.org/pdf/2208.09466v1)

**Authors**: Vyas Raina, Mark Gales

**Abstracts**: Grammatical Error Correction (GEC) systems perform a sequence-to-sequence task, where an input word sequence containing grammatical errors, is corrected for these errors by the GEC system to output a grammatically correct word sequence. With the advent of deep learning methods, automated GEC systems have become increasingly popular. For example, GEC systems are often used on speech transcriptions of English learners as a form of assessment and feedback - these powerful GEC systems can be used to automatically measure an aspect of a candidate's fluency. The count of \textit{edits} from a candidate's input sentence (or essay) to a GEC system's grammatically corrected output sentence is indicative of a candidate's language ability, where fewer edits suggest better fluency. The count of edits can thus be viewed as a \textit{fluency score} with zero implying perfect fluency. However, although deep learning based GEC systems are extremely powerful and accurate, they are susceptible to adversarial attacks: an adversary can introduce a small, specific change at the input of a system that causes a large, undesired change at the output. When considering the application of GEC systems to automated language assessment, the aim of an adversary could be to cheat by making a small change to a grammatically incorrect input sentence that conceals the errors from a GEC system, such that no edits are found and the candidate is unjustly awarded a perfect fluency score. This work examines a simple universal substitution adversarial attack that non-native speakers of English could realistically employ to deceive GEC systems used for assessment.



## **49. Curbing Task Interference using Representation Similarity-Guided Multi-Task Feature Sharing**

cs.CV

Published at 1st Conference on Lifelong Learning Agents (CoLLAs 2022)

**SubmitDate**: 2022-08-19    [paper-pdf](http://arxiv.org/pdf/2208.09427v1)

**Authors**: Naresh Kumar Gurulingan, Elahe Arani, Bahram Zonooz

**Abstracts**: Multi-task learning of dense prediction tasks, by sharing both the encoder and decoder, as opposed to sharing only the encoder, provides an attractive front to increase both accuracy and computational efficiency. When the tasks are similar, sharing the decoder serves as an additional inductive bias providing more room for tasks to share complementary information among themselves. However, increased sharing exposes more parameters to task interference which likely hinders both generalization and robustness. Effective ways to curb this interference while exploiting the inductive bias of sharing the decoder remains an open challenge. To address this challenge, we propose Progressive Decoder Fusion (PDF) to progressively combine task decoders based on inter-task representation similarity. We show that this procedure leads to a multi-task network with better generalization to in-distribution and out-of-distribution data and improved robustness to adversarial attacks. Additionally, we observe that the predictions of different tasks of this multi-task network are more consistent with each other.



## **50. A Pragmatic Methodology for Blind Hardware Trojan Insertion in Finalized Layouts**

cs.CR

9 pages, 6 figures, 3 tables, to be published in ICCAD 2022

**SubmitDate**: 2022-08-19    [paper-pdf](http://arxiv.org/pdf/2208.09235v1)

**Authors**: Alexander Hepp, Tiago Perez, Samuel Pagliarini, Georg Sigl

**Abstracts**: A potential vulnerability for integrated circuits (ICs) is the insertion of hardware trojans (HTs) during manufacturing. Understanding the practicability of such an attack can lead to appropriate measures for mitigating it. In this paper, we demonstrate a pragmatic framework for analyzing HT susceptibility of finalized layouts. Our framework is representative of a fabrication-time attack, where the adversary is assumed to have access only to a layout representation of the circuit. The framework inserts trojans into tapeout-ready layouts utilizing an Engineering Change Order (ECO) flow. The attacked security nodes are blindly searched utilizing reverse-engineering techniques. For our experimental investigation, we utilized three crypto-cores (AES-128, SHA-256, and RSA) and a microcontroller (RISC-V) as targets. We explored 96 combinations of triggers, payloads and targets for our framework. Our findings demonstrate that even in high-density designs, the covert insertion of sophisticated trojans is possible. All this while maintaining the original target logic, with minimal impact on power and performance. Furthermore, from our exploration, we conclude that it is too naive to only utilize placement resources as a metric for HT vulnerability. This work highlights that the HT insertion success is a complex function of the placement, routing resources, the position of the attacked nodes, and further design-specific characteristics. As a result, our framework goes beyond just an attack, we present the most advanced analysis tool to assess the vulnerability of HT insertion into finalized layouts.



