# Latest Adversarial Attack Papers
**update at 2023-02-16 15:59:47**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. XploreNAS: Explore Adversarially Robust & Hardware-efficient Neural Architectures for Non-ideal Xbars**

cs.LG

16 pages, 8 figures, 2 tables

**SubmitDate**: 2023-02-15    [abs](http://arxiv.org/abs/2302.07769v1) [paper-pdf](http://arxiv.org/pdf/2302.07769v1)

**Authors**: Abhiroop Bhattacharjee, Abhishek Moitra, Priyadarshini Panda

**Abstract**: Compute In-Memory platforms such as memristive crossbars are gaining focus as they facilitate acceleration of Deep Neural Networks (DNNs) with high area and compute-efficiencies. However, the intrinsic non-idealities associated with the analog nature of computing in crossbars limits the performance of the deployed DNNs. Furthermore, DNNs are shown to be vulnerable to adversarial attacks leading to severe security threats in their large-scale deployment. Thus, finding adversarially robust DNN architectures for non-ideal crossbars is critical to the safe and secure deployment of DNNs on the edge. This work proposes a two-phase algorithm-hardware co-optimization approach called XploreNAS that searches for hardware-efficient & adversarially robust neural architectures for non-ideal crossbar platforms. We use the one-shot Neural Architecture Search (NAS) approach to train a large Supernet with crossbar-awareness and sample adversarially robust Subnets therefrom, maintaining competitive hardware-efficiency. Our experiments on crossbars with benchmark datasets (SVHN, CIFAR10 & CIFAR100) show upto ~8-16% improvement in the adversarial robustness of the searched Subnets against a baseline ResNet-18 model subjected to crossbar-aware adversarial training. We benchmark our robust Subnets for Energy-Delay-Area-Products (EDAPs) using the Neurosim tool and find that with additional hardware-efficiency driven optimizations, the Subnets attain ~1.5-1.6x lower EDAPs than ResNet-18 baseline.



## **2. Quantum key distribution with post-processing driven by physical unclonable functions**

quant-ph

**SubmitDate**: 2023-02-15    [abs](http://arxiv.org/abs/2302.07623v1) [paper-pdf](http://arxiv.org/pdf/2302.07623v1)

**Authors**: Georgios M. Nikolopoulos, Marc Fischlin

**Abstract**: Quantum key-distribution protocols allow two honest distant parties to establish a common truly random secret key in the presence of powerful adversaries, provided that the two users share beforehand a short secret key. This pre-shared secret key is used mainly for authentication purposes in the post-processing of classical data that have been obtained during the quantum communication stage, and it prevents a man-in-the-middle attack. The necessity of a pre-shared key is usually considered as the main drawback of quantum key-distribution protocols, which becomes even stronger for large networks involving more that two users. Here we discuss the conditions under which physical unclonable function can be integrated in currently available quantum key-distribution systems, in order to facilitate the generation and the distribution of the necessary pre-shared key, with the smallest possible cost in the security of the systems. Moreover, the integration of physical unclonable functions in quantum key-distribution networks allows for real-time authentication of the devices that are connected to the network.



## **3. 3D-VFD: A Victim-free Detector against 3D Adversarial Point Clouds**

cs.MM

6 pages, 13pages

**SubmitDate**: 2023-02-15    [abs](http://arxiv.org/abs/2205.08738v3) [paper-pdf](http://arxiv.org/pdf/2205.08738v3)

**Authors**: Jiahao Zhu, Huajun Zhou, Zixuan Chen, Yi Zhou, Xiaohua Xie

**Abstract**: 3D deep models consuming point clouds have achieved sound application effects in computer vision. However, recent studies have shown they are vulnerable to 3D adversarial point clouds. In this paper, we regard these malicious point clouds as 3D steganography examples and present a new perspective, 3D steganalysis, to counter such examples. Specifically, we propose 3D-VFD, a victim-free detector against 3D adversarial point clouds. Its core idea is to capture the discrepancies between residual geometric feature distributions of benign point clouds and adversarial point clouds and map these point clouds to a lower dimensional space where we can efficiently distinguish them. Unlike existing detection techniques against 3D adversarial point clouds, 3D-VFD does not rely on the victim 3D deep model's outputs for discrimination. Extensive experiments demonstrate that 3D-VFD achieves state-of-the-art detection and can effectively detect 3D adversarial attacks based on point adding and point perturbation while keeping fast detection speed.



## **4. Attacking Fake News Detectors via Manipulating News Social Engagement**

cs.SI

In Proceedings of the ACM Web Conference 2023 (WWW'23)

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2302.07363v1) [paper-pdf](http://arxiv.org/pdf/2302.07363v1)

**Authors**: Haoran Wang, Yingtong Dou, Canyu Chen, Lichao Sun, Philip S. Yu, Kai Shu

**Abstract**: Social media is one of the main sources for news consumption, especially among the younger generation. With the increasing popularity of news consumption on various social media platforms, there has been a surge of misinformation which includes false information or unfounded claims. As various text- and social context-based fake news detectors are proposed to detect misinformation on social media, recent works start to focus on the vulnerabilities of fake news detectors. In this paper, we present the first adversarial attack framework against Graph Neural Network (GNN)-based fake news detectors to probe their robustness. Specifically, we leverage a multi-agent reinforcement learning (MARL) framework to simulate the adversarial behavior of fraudsters on social media. Research has shown that in real-world settings, fraudsters coordinate with each other to share different news in order to evade the detection of fake news detectors. Therefore, we modeled our MARL framework as a Markov Game with bot, cyborg, and crowd worker agents, which have their own distinctive cost, budget, and influence. We then use deep Q-learning to search for the optimal policy that maximizes the rewards. Extensive experimental results on two real-world fake news propagation datasets demonstrate that our proposed framework can effectively sabotage the GNN-based fake news detector performance. We hope this paper can provide insights for future research on fake news detection.



## **5. Cooperative Perception for Safe Control of Autonomous Vehicles under LiDAR Spoofing Attacks**

eess.SY

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2302.07341v1) [paper-pdf](http://arxiv.org/pdf/2302.07341v1)

**Authors**: Hongchao Zhang, Zhouchi Li, Shiyu Cheng, Andrew Clark

**Abstract**: Autonomous vehicles rely on LiDAR sensors to detect obstacles such as pedestrians, other vehicles, and fixed infrastructures. LiDAR spoofing attacks have been demonstrated that either create erroneous obstacles or prevent detection of real obstacles, resulting in unsafe driving behaviors. In this paper, we propose an approach to detect and mitigate LiDAR spoofing attacks by leveraging LiDAR scan data from other neighboring vehicles. This approach exploits the fact that spoofing attacks can typically only be mounted on one vehicle at a time, and introduce additional points into the victim's scan that can be readily detected by comparison from other, non-modified scans. We develop a Fault Detection, Identification, and Isolation procedure that identifies non-existing obstacle, physical removal, and adversarial object attacks, while also estimating the actual locations of obstacles. We propose a control algorithm that guarantees that these estimated object locations are avoided. We validate our framework using a CARLA simulation study, in which we verify that our FDII algorithm correctly detects each attack pattern.



## **6. Randomization for adversarial robustness: the Good, the Bad and the Ugly**

cs.LG

8 pages + bibliography and appendix, 3 figures. Submitted to ICML  2023

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2302.07221v1) [paper-pdf](http://arxiv.org/pdf/2302.07221v1)

**Authors**: Lucas Gnecco-Heredia, Yann Chevaleyre, Benjamin Negrevergne, Laurent Meunier

**Abstract**: Deep neural networks are known to be vulnerable to adversarial attacks: A small perturbation that is imperceptible to a human can easily make a well-trained deep neural network misclassify. To defend against adversarial attacks, randomized classifiers have been proposed as a robust alternative to deterministic ones. In this work we show that in the binary classification setting, for any randomized classifier, there is always a deterministic classifier with better adversarial risk. In other words, randomization is not necessary for robustness. In many common randomization schemes, the deterministic classifiers with better risk are explicitly described: For example, we show that ensembles of classifiers are more robust than mixtures of classifiers, and randomized smoothing is more robust than input noise injection. Finally, experiments confirm our theoretical results with the two families of randomized classifiers we analyze.



## **7. Bridge the Gap Between CV and NLP! An Optimization-based Textual Adversarial Attack Framework**

cs.CL

Codes are available at: https://github.com/Phantivia/T-PGD

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2110.15317v3) [paper-pdf](http://arxiv.org/pdf/2110.15317v3)

**Authors**: Lifan Yuan, Yichi Zhang, Yangyi Chen, Wei Wei

**Abstract**: Despite recent success on various tasks, deep learning techniques still perform poorly on adversarial examples with small perturbations. While optimization-based methods for adversarial attacks are well-explored in the field of computer vision, it is impractical to directly apply them in natural language processing due to the discrete nature of the text. To address the problem, we propose a unified framework to extend the existing optimization-based adversarial attack methods in the vision domain to craft textual adversarial samples. In this framework, continuously optimized perturbations are added to the embedding layer and amplified in the forward propagation process. Then the final perturbed latent representations are decoded with a masked language model head to obtain potential adversarial samples. In this paper, we instantiate our framework with an attack algorithm named Textual Projected Gradient Descent (T-PGD). We find our algorithm effective even using proxy gradient information. Therefore, we perform the more challenging transfer black-box attack and conduct comprehensive experiments to evaluate our attack algorithm with several models on three benchmark datasets. Experimental results demonstrate that our method achieves an overall better performance and produces more fluent and grammatical adversarial samples compared to strong baseline methods. All the code and data will be made public.



## **8. A Comprehensive Study of Real-Time Object Detection Networks Across Multiple Domains: A Survey**

cs.CV

Published in Transactions on Machine Learning Research (TMLR) with  Survey Certification

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2208.10895v2) [paper-pdf](http://arxiv.org/pdf/2208.10895v2)

**Authors**: Elahe Arani, Shruthi Gowda, Ratnajit Mukherjee, Omar Magdy, Senthilkumar Kathiresan, Bahram Zonooz

**Abstract**: Deep neural network based object detectors are continuously evolving and are used in a multitude of applications, each having its own set of requirements. While safety-critical applications need high accuracy and reliability, low-latency tasks need resource and energy-efficient networks. Real-time detectors, which are a necessity in high-impact real-world applications, are continuously proposed, but they overemphasize the improvements in accuracy and speed while other capabilities such as versatility, robustness, resource and energy efficiency are omitted. A reference benchmark for existing networks does not exist, nor does a standard evaluation guideline for designing new networks, which results in ambiguous and inconsistent comparisons. We, thus, conduct a comprehensive study on multiple real-time detectors (anchor-, keypoint-, and transformer-based) on a wide range of datasets and report results on an extensive set of metrics. We also study the impact of variables such as image size, anchor dimensions, confidence thresholds, and architecture layers on the overall performance. We analyze the robustness of detection networks against distribution shifts, natural corruptions, and adversarial attacks. Also, we provide a calibration analysis to gauge the reliability of the predictions. Finally, to highlight the real-world impact, we conduct two unique case studies, on autonomous driving and healthcare applications. To further gauge the capability of networks in critical real-time applications, we report the performance after deploying the detection networks on edge devices. Our extensive empirical study can act as a guideline for the industrial community to make an informed choice on the existing networks. We also hope to inspire the research community towards a new direction in the design and evaluation of networks that focuses on a bigger and holistic overview for a far-reaching impact.



## **9. Practical Cross-system Shilling Attacks with Limited Access to Data**

cs.IR

Accepted by AAAI 2023

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2302.07145v1) [paper-pdf](http://arxiv.org/pdf/2302.07145v1)

**Authors**: Meifang Zeng, Ke Li, Bingchuan Jiang, Liujuan Cao, Hui Li

**Abstract**: In shilling attacks, an adversarial party injects a few fake user profiles into a Recommender System (RS) so that the target item can be promoted or demoted. Although much effort has been devoted to developing shilling attack methods, we find that existing approaches are still far from practical. In this paper, we analyze the properties a practical shilling attack method should have and propose a new concept of Cross-system Attack. With the idea of Cross-system Attack, we design a Practical Cross-system Shilling Attack (PC-Attack) framework that requires little information about the victim RS model and the target RS data for conducting attacks. PC-Attack is trained to capture graph topology knowledge from public RS data in a self-supervised manner. Then, it is fine-tuned on a small portion of target data that is easy to access to construct fake profiles. Extensive experiments have demonstrated the superiority of PC-Attack over state-of-the-art baselines. Our implementation of PC-Attack is available at https://github.com/KDEGroup/PC-Attack.



## **10. Adversarial Path Planning for Optimal Camera Positioning**

cs.CG

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2302.07051v1) [paper-pdf](http://arxiv.org/pdf/2302.07051v1)

**Authors**: Gaia Carenini, Alexandre Duplessis

**Abstract**: The use of visual sensors is flourishing, driven among others by the several applications in detection and prevention of crimes or dangerous events. While the problem of optimal camera placement for total coverage has been solved for a decade or so, that of the arrangement of cameras maximizing the recognition of objects "in-transit" is still open. The objective of this paper is to attack this problem by providing an adversarial method of proven optimality based on the resolution of Hamilton-Jacobi equations. The problem is attacked by first assuming the perspective of an adversary, i.e. computing explicitly the path minimizing the probability of detection and the quality of reconstruction. Building on this result, we introduce an optimality measure for camera configurations and perform a simulated annealing algorithm to find the optimal camera placement.



## **11. Does CLIP Know My Face?**

cs.LG

15 pages, 6 figures

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2209.07341v2) [paper-pdf](http://arxiv.org/pdf/2209.07341v2)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Manuel Brack, Felix Friedrich, Patrick Schramowski, Kristian Kersting

**Abstract**: With the rise of deep learning in various applications, privacy concerns around the protection of training data has become a critical area of research. Whereas prior studies have focused on privacy risks in single-modal models, we introduce a novel method to assess privacy for multi-modal models, specifically vision-language models like CLIP. The proposed Identity Inference Attack (IDIA) reveals whether an individual was included in the training data by querying the model with images of the same person. Letting the model choose from a wide variety of possible text labels, the model reveals whether it recognizes the person and, therefore, was used for training. Our large-scale experiments on CLIP demonstrate that individuals used for training can be identified with very high accuracy. We confirm that the model has learned to associate names with depicted individuals, implying the existence of sensitive information that can be extracted by adversaries. Our results highlight the need for stronger privacy protection in large-scale models and suggest that IDIAs can be used to prove the unauthorized use of data for training and to enforce privacy laws.



## **12. Fact-Saboteurs: A Taxonomy of Evidence Manipulation Attacks against Fact-Verification Systems**

cs.CR

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2209.03755v3) [paper-pdf](http://arxiv.org/pdf/2209.03755v3)

**Authors**: Sahar Abdelnabi, Mario Fritz

**Abstract**: Mis- and disinformation are a substantial global threat to our security and safety. To cope with the scale of online misinformation, researchers have been working on automating fact-checking by retrieving and verifying against relevant evidence. However, despite many advances, a comprehensive evaluation of the possible attack vectors against such systems is still lacking. Particularly, the automated fact-verification process might be vulnerable to the exact disinformation campaigns it is trying to combat. In this work, we assume an adversary that automatically tampers with the online evidence in order to disrupt the fact-checking model via camouflaging the relevant evidence or planting a misleading one. We first propose an exploratory taxonomy that spans these two targets and the different threat model dimensions. Guided by this, we design and propose several potential attack methods. We show that it is possible to subtly modify claim-salient snippets in the evidence and generate diverse and claim-aligned evidence. Thus, we highly degrade the fact-checking performance under many different permutations of the taxonomy's dimensions. The attacks are also robust against post-hoc modifications of the claim. Our analysis further hints at potential limitations in models' inference when faced with contradicting evidence. We emphasize that these attacks can have harmful implications on the inspectable and human-in-the-loop usage scenarios of such models, and conclude by discussing challenges and directions for future defenses.



## **13. Oops..! I Glitched It Again! How to Multi-Glitch the Glitching-Protections on ARM TrustZone-M**

cs.CR

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2302.06932v1) [paper-pdf](http://arxiv.org/pdf/2302.06932v1)

**Authors**: Marvin Saß, Richard Mitev, Ahmad-Reza Sadeghi

**Abstract**: Voltage Fault Injection (VFI), also known as power glitching, has proven to be a severe threat to real-world systems. In VFI attacks, the adversary disturbs the power-supply of the target-device forcing the device to illegitimate behavior. Various countermeasures have been proposed to address different types of fault injection attacks at different abstraction layers, either requiring to modify the underlying hardware or software/firmware at the machine instruction level. Moreover, only recently, individual chip manufacturers have started to respond to this threat by integrating countermeasures in their products. Generally, these countermeasures aim at protecting against single fault injection (SFI) attacks, since Multiple Fault Injection (MFI) is believed to be challenging and sometimes even impractical. In this paper, we present {\mu}-Glitch, the first Voltage Fault Injection (VFI) platform which is capable of injecting multiple, coordinated voltage faults into a target device, requiring only a single trigger signal. We provide a novel flow for Multiple Voltage Fault Injection (MVFI) attacks to significantly reduce the search complexity for fault parameters, as the search space increases exponentially with each additional fault injection. We evaluate and showcase the effectiveness and practicality of our attack platform on four real-world chips, featuring TrustZone-M: The first two have interdependent backchecking mechanisms, while the second two have additionally integrated countermeasures against fault injection. Our evaluation revealed that {\mu}-Glitch can successfully inject four consecutive faults within an average time of one day. Finally, we discuss potential countermeasures to mitigate VFI attacks and additionally propose two novel attack scenarios for MVFI.



## **14. Towards Robust Neural Image Compression: Adversarial Attack and Model Finetuning**

cs.CV

This paper has been completely rewritten

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2112.08691v2) [paper-pdf](http://arxiv.org/pdf/2112.08691v2)

**Authors**: Tong Chen, Zhan Ma

**Abstract**: Deep neural network based image compression has been extensively studied. Model robustness is largely overlooked, though it is crucial to service enabling. We perform the adversarial attack by injecting a small amount of noise perturbation to original source images, and then encode these adversarial examples using prevailing learnt image compression models. Experiments report severe distortion in the reconstruction of adversarial examples, revealing the general vulnerability of existing methods, regardless of the settings used in underlying compression model (e.g., network architecture, loss function, quality scale) and optimization strategy used for injecting perturbation (e.g., noise threshold, signal distance measurement). Later, we apply the iterative adversarial finetuning to refine pretrained models. In each iteration, random source images and adversarial examples are mixed to update underlying model. Results show the effectiveness of the proposed finetuning strategy by substantially improving the compression model robustness. Overall, our methodology is simple, effective, and generalizable, making it attractive for developing robust learnt image compression solution. All materials have been made publicly accessible at https://njuvision.github.io/RobustNIC for reproducible research.



## **15. Towards Effective and Robust Neural Trojan Defenses via Input Filtering**

cs.CR

Accepted to ECCV 2022

**SubmitDate**: 2023-02-14    [abs](http://arxiv.org/abs/2202.12154v5) [paper-pdf](http://arxiv.org/pdf/2202.12154v5)

**Authors**: Kien Do, Haripriya Harikumar, Hung Le, Dung Nguyen, Truyen Tran, Santu Rana, Dang Nguyen, Willy Susilo, Svetha Venkatesh

**Abstract**: Trojan attacks on deep neural networks are both dangerous and surreptitious. Over the past few years, Trojan attacks have advanced from using only a single input-agnostic trigger and targeting only one class to using multiple, input-specific triggers and targeting multiple classes. However, Trojan defenses have not caught up with this development. Most defense methods still make inadequate assumptions about Trojan triggers and target classes, thus, can be easily circumvented by modern Trojan attacks. To deal with this problem, we propose two novel "filtering" defenses called Variational Input Filtering (VIF) and Adversarial Input Filtering (AIF) which leverage lossy data compression and adversarial learning respectively to effectively purify potential Trojan triggers in the input at run time without making assumptions about the number of triggers/target classes or the input dependence property of triggers. In addition, we introduce a new defense mechanism called "Filtering-then-Contrasting" (FtC) which helps avoid the drop in classification accuracy on clean data caused by "filtering", and combine it with VIF/AIF to derive new defenses of this kind. Extensive experimental results and ablation studies show that our proposed defenses significantly outperform well-known baseline defenses in mitigating five advanced Trojan attacks including two recent state-of-the-art while being quite robust to small amounts of training data and large-norm triggers.



## **16. A survey in Adversarial Defences and Robustness in NLP**

cs.CL

**SubmitDate**: 2023-02-13    [abs](http://arxiv.org/abs/2203.06414v3) [paper-pdf](http://arxiv.org/pdf/2203.06414v3)

**Authors**: Shreya Goyal, Sumanth Doddapaneni, Mitesh M. Khapra, Balaraman Ravindran

**Abstract**: In recent years, it has been seen that deep neural networks are lacking robustness and are vulnerable in case of adversarial perturbations in input data. Strong adversarial attacks are proposed by various authors for tasks under computer vision and Natural Language Processing (NLP). As a counter-effort, several defense mechanisms are also proposed to save these networks from failing. Defending the neural networks from adversarial attacks has its own importance, where the goal is to ensure that the model's prediction doesn't change if input data is perturbed. Numerous methods for adversarial defense in NLP are proposed of late, for different NLP tasks such as text classification, named entity recognition, natural language inferencing, etc. Some of these methods are not just used for defending neural networks from adversarial attacks, but also used as a regularization mechanism during training, saving the model from overfitting. The proposed survey is an attempt to review different methods proposed for adversarial defenses in NLP in recent years by proposing a novel taxonomy. This survey also highlights the fragility of the advanced deep neural networks in NLP and the challenges in defending them.



## **17. Sneaky Spikes: Uncovering Stealthy Backdoor Attacks in Spiking Neural Networks with Neuromorphic Data**

cs.CR

**SubmitDate**: 2023-02-13    [abs](http://arxiv.org/abs/2302.06279v1) [paper-pdf](http://arxiv.org/pdf/2302.06279v1)

**Authors**: Gorka Abad, Oguzhan Ersoy, Stjepan Picek, Aitor Urbieta

**Abstract**: Deep neural networks (DNNs) have achieved excellent results in various tasks, including image and speech recognition. However, optimizing the performance of DNNs requires careful tuning of multiple hyperparameters and network parameters via training. High-performance DNNs utilize a large number of parameters, corresponding to high energy consumption during training. To address these limitations, researchers have developed spiking neural networks (SNNs), which are more energy-efficient and can process data in a biologically plausible manner, making them well-suited for tasks involving sensory data processing, i.e., neuromorphic data. Like DNNs, SNNs are vulnerable to various threats, such as adversarial examples and backdoor attacks. Yet, the attacks and countermeasures for SNNs have been almost fully unexplored.   This paper investigates the application of backdoor attacks in SNNs using neuromorphic datasets and different triggers. More precisely, backdoor triggers in neuromorphic data can change their position and color, allowing a larger range of possibilities than common triggers in, e.g., the image domain. We propose different attacks achieving up to 100\% attack success rate without noticeable clean accuracy degradation. We also evaluate the stealthiness of the attacks via the structural similarity metric, showing our most powerful attacks being also stealthy. Finally, we adapt the state-of-the-art defenses from the image domain, demonstrating they are not necessarily effective for neuromorphic data resulting in inaccurate performance.



## **18. PRAGTHOS:Practical Game Theoretically Secure Proof-of-Work Blockchain**

cs.CR

**SubmitDate**: 2023-02-13    [abs](http://arxiv.org/abs/2302.06136v1) [paper-pdf](http://arxiv.org/pdf/2302.06136v1)

**Authors**: Varul Srivastava, Dr. Sujit Gujar

**Abstract**: Security analysis of blockchain technology is an active domain of research. There has been both cryptographic and game-theoretic security analysis of Proof-of-Work (PoW) blockchains. Prominent work includes the cryptographic security analysis under the Universal Composable framework and Game-theoretic security analysis using Rational Protocol Design. These security analysis models rely on stricter assumptions that might not hold. In this paper, we analyze the security of PoW blockchain protocols. We first show how assumptions made by previous models need not be valid in reality, which attackers can exploit to launch attacks that these models fail to capture. These include Difficulty Alternating Attack, under which forking is possible for an adversary with less than 0.5 mining power, Quick-Fork Attack, a general bound on selfish mining attack and transaction withholding attack. Following this, we argue why previous models for security analysis fail to capture these attacks and propose a more practical framework for security analysis pRPD. We then propose a framework to build PoW blockchains PRAGTHOS, which is secure from the attacks mentioned above. Finally, we argue that PoW blockchains complying with the PRAGTHOS framework are secure against a computationally bounded adversary under certain conditions on the reward scheme.



## **19. GAIN: Enhancing Byzantine Robustness in Federated Learning with Gradient Decomposition**

cs.LG

**SubmitDate**: 2023-02-13    [abs](http://arxiv.org/abs/2302.06079v1) [paper-pdf](http://arxiv.org/pdf/2302.06079v1)

**Authors**: Yuchen Liu, Chen Chen, Lingjuan Lyu, Fangzhao Wu, Sai Wu, Gang Chen

**Abstract**: Federated learning provides a privacy-aware learning framework by enabling participants to jointly train models without exposing their private data. However, federated learning has exhibited vulnerabilities to Byzantine attacks, where the adversary aims to destroy the convergence and performance of the global model. Meanwhile, we observe that most existing robust AGgregation Rules (AGRs) fail to stop the aggregated gradient deviating from the optimal gradient (the average of honest gradients) in the non-IID setting. We attribute the reason of the failure of these AGRs to two newly proposed concepts: identification failure and integrity failure. The identification failure mainly comes from the exacerbated curse of dimensionality in the non-IID setting. The integrity failure is a combined result of conservative filtering strategy and gradient heterogeneity. In order to address both failures, we propose GAIN, a gradient decomposition scheme that can help adapt existing robust algorithms to heterogeneous datasets. We also provide convergence analysis for integrating existing robust AGRs into GAIN. Experiments on various real-world datasets verify the efficacy of our proposed GAIN.



## **20. An Integrated Approach to Produce Robust Models with High Efficiency**

cs.CV

**SubmitDate**: 2023-02-13    [abs](http://arxiv.org/abs/2008.13305v4) [paper-pdf](http://arxiv.org/pdf/2008.13305v4)

**Authors**: Zhijian Li, Bao Wang, Jack Xin

**Abstract**: Deep Neural Networks (DNNs) needs to be both efficient and robust for practical uses. Quantization and structure simplification are promising ways to adapt DNNs to mobile devices, and adversarial training is the most popular method to make DNNs robust. In this work, we try to obtain both features by applying a convergent relaxation quantization algorithm, Binary-Relax (BR), to a robust adversarial-trained model, ResNets Ensemble via Feynman-Kac Formalism (EnResNet). We also discover that high precision, such as ternary (tnn) and 4-bit, quantization will produce sparse DNNs. However, this sparsity is unstructured under advarsarial training. To solve the problems that adversarial training jeopardizes DNNs' accuracy on clean images and the struture of sparsity, we design a trade-off loss function that helps DNNs preserve their natural accuracy and improve the channel sparsity. With our trade-off loss function, we achieve both goals with no reduction of resistance under weak attacks and very minor reduction of resistance under strong attcks. Together with quantized EnResNet with trade-off loss function, we provide robust models that have high efficiency.



## **21. TextDefense: Adversarial Text Detection based on Word Importance Entropy**

cs.CL

**SubmitDate**: 2023-02-12    [abs](http://arxiv.org/abs/2302.05892v1) [paper-pdf](http://arxiv.org/pdf/2302.05892v1)

**Authors**: Lujia Shen, Xuhong Zhang, Shouling Ji, Yuwen Pu, Chunpeng Ge, Xing Yang, Yanghe Feng

**Abstract**: Currently, natural language processing (NLP) models are wildly used in various scenarios. However, NLP models, like all deep models, are vulnerable to adversarially generated text. Numerous works have been working on mitigating the vulnerability from adversarial attacks. Nevertheless, there is no comprehensive defense in existing works where each work targets a specific attack category or suffers from the limitation of computation overhead, irresistible to adaptive attack, etc.   In this paper, we exhaustively investigate the adversarial attack algorithms in NLP, and our empirical studies have discovered that the attack algorithms mainly disrupt the importance distribution of words in a text. A well-trained model can distinguish subtle importance distribution differences between clean and adversarial texts. Based on this intuition, we propose TextDefense, a new adversarial example detection framework that utilizes the target model's capability to defend against adversarial attacks while requiring no prior knowledge. TextDefense differs from previous approaches, where it utilizes the target model for detection and thus is attack type agnostic. Our extensive experiments show that TextDefense can be applied to different architectures, datasets, and attack methods and outperforms existing methods. We also discover that the leading factor influencing the performance of TextDefense is the target model's generalizability. By analyzing the property of the target model and the property of the adversarial example, we provide our insights into the adversarial attacks in NLP and the principles of our defense method.



## **22. Mutation-Based Adversarial Attacks on Neural Text Detectors**

cs.CR

**SubmitDate**: 2023-02-11    [abs](http://arxiv.org/abs/2302.05794v1) [paper-pdf](http://arxiv.org/pdf/2302.05794v1)

**Authors**: Gongbo Liang, Jesus Guerrero, Izzat Alsmadi

**Abstract**: Neural text detectors aim to decide the characteristics that distinguish neural (machine-generated) from human texts. To challenge such detectors, adversarial attacks can alter the statistical characteristics of the generated text, making the detection task more and more difficult. Inspired by the advances of mutation analysis in software development and testing, in this paper, we propose character- and word-based mutation operators for generating adversarial samples to attack state-of-the-art natural text detectors. This falls under white-box adversarial attacks. In such attacks, attackers have access to the original text and create mutation instances based on this original text. The ultimate goal is to confuse machine learning models and classifiers and decrease their prediction accuracy.



## **23. Escaping saddle points in zeroth-order optimization: the power of two-point estimators**

math.OC

This new version includes an improved sample complexity result for  strict saddle functions, new simulation results, an updated introduction, as  well as more streamlined proof outlines

**SubmitDate**: 2023-02-11    [abs](http://arxiv.org/abs/2209.13555v2) [paper-pdf](http://arxiv.org/pdf/2209.13555v2)

**Authors**: Zhaolin Ren, Yujie Tang, Na Li

**Abstract**: Two-point zeroth order methods are important in many applications of zeroth-order optimization, such as robotics, wind farms, power systems, online optimization, and adversarial robustness to black-box attacks in deep neural networks, where the problem may be high-dimensional and/or time-varying. Most problems in these applications are nonconvex and contain saddle points. While existing works have shown that zeroth-order methods utilizing $\Omega(d)$ function valuations per iteration (with $d$ denoting the problem dimension) can escape saddle points efficiently, it remains an open question if zeroth-order methods based on two-point estimators can escape saddle points. In this paper, we show that by adding an appropriate isotropic perturbation at each iteration, a zeroth-order algorithm based on $2m$ (for any $1 \leq m \leq d$) function evaluations per iteration can not only find $\epsilon$-second order stationary points polynomially fast, but do so using only $\tilde{O}\left(\frac{d}{\epsilon^{2}\bar{\psi}}\right)$ function evaluations, where $\bar{\psi} \geq \tilde{\Omega}(\sqrt{\epsilon})$ is a parameter capturing the extent to which the function of interest exhibits the strict saddle property.



## **24. Exploiting Programmatic Behavior of LLMs: Dual-Use Through Standard Security Attacks**

cs.CR

**SubmitDate**: 2023-02-11    [abs](http://arxiv.org/abs/2302.05733v1) [paper-pdf](http://arxiv.org/pdf/2302.05733v1)

**Authors**: Daniel Kang, Xuechen Li, Ion Stoica, Carlos Guestrin, Matei Zaharia, Tatsunori Hashimoto

**Abstract**: Recent advances in instruction-following large language models (LLMs) have led to dramatic improvements in a range of NLP tasks. Unfortunately, we find that the same improved capabilities amplify the dual-use risks for malicious purposes of these models. Dual-use is difficult to prevent as instruction-following capabilities now enable standard attacks from computer security. The capabilities of these instruction-following LLMs provide strong economic incentives for dual-use by malicious actors. In particular, we show that instruction-following LLMs can produce targeted malicious content, including hate speech and scams, bypassing in-the-wild defenses implemented by LLM API vendors. Our analysis shows that this content can be generated economically and at cost likely lower than with human effort alone. Together, our findings suggest that LLMs will increasingly attract more sophisticated adversaries and attacks, and addressing these attacks may require new approaches to mitigations.



## **25. HateProof: Are Hateful Meme Detection Systems really Robust?**

cs.CL

Accepted at TheWebConf'2023 (WWW'2023)

**SubmitDate**: 2023-02-11    [abs](http://arxiv.org/abs/2302.05703v1) [paper-pdf](http://arxiv.org/pdf/2302.05703v1)

**Authors**: Piush Aggarwal, Pranit Chawla, Mithun Das, Punyajoy Saha, Binny Mathew, Torsten Zesch, Animesh Mukherjee

**Abstract**: Exploiting social media to spread hate has tremendously increased over the years. Lately, multi-modal hateful content such as memes has drawn relatively more traction than uni-modal content. Moreover, the availability of implicit content payloads makes them fairly challenging to be detected by existing hateful meme detection systems. In this paper, we present a use case study to analyze such systems' vulnerabilities against external adversarial attacks. We find that even very simple perturbations in uni-modal and multi-modal settings performed by humans with little knowledge about the model can make the existing detection models highly vulnerable. Empirically, we find a noticeable performance drop of as high as 10% in the macro-F1 score for certain attacks. As a remedy, we attempt to boost the model's robustness using contrastive learning as well as an adversarial training-based method - VILLA. Using an ensemble of the above two approaches, in two of our high resolution datasets, we are able to (re)gain back the performance to a large extent for certain attacks. We believe that ours is a first step toward addressing this crucial problem in an adversarial setting and would inspire more such investigations in the future.



## **26. High Recovery with Fewer Injections: Practical Binary Volumetric Injection Attacks against Dynamic Searchable Encryption**

cs.CR

22 pages, 19 fugures, will be published in USENIX Security 2023

**SubmitDate**: 2023-02-11    [abs](http://arxiv.org/abs/2302.05628v1) [paper-pdf](http://arxiv.org/pdf/2302.05628v1)

**Authors**: Xianglong Zhang, Wei Wang, Peng Xu, Laurence T. Yang, Kaitai Liang

**Abstract**: Searchable symmetric encryption enables private queries over an encrypted database, but it also yields information leakages. Adversaries can exploit these leakages to launch injection attacks (Zhang et al., USENIX'16) to recover the underlying keywords from queries. The performance of the existing injection attacks is strongly dependent on the amount of leaked information or injection. In this work, we propose two new injection attacks, namely BVA and BVMA, by leveraging a binary volumetric approach. We enable adversaries to inject fewer files than the existing volumetric attacks by using the known keywords and reveal the queries by observing the volume of the query results. Our attacks can thwart well-studied defenses (e.g., threshold countermeasure, static padding) without exploiting the distribution of target queries and client databases. We evaluate the proposed attacks empirically in real-world datasets with practical queries. The results show that our attacks can obtain a high recovery rate (>80%) in the best case and a roughly 60% recovery even under a large-scale dataset with a small number of injections (<20 files).



## **27. Towards A Proactive ML Approach for Detecting Backdoor Poison Samples**

cs.LG

**SubmitDate**: 2023-02-10    [abs](http://arxiv.org/abs/2205.13616v2) [paper-pdf](http://arxiv.org/pdf/2205.13616v2)

**Authors**: Xiangyu Qi, Tinghao Xie, Jiachen T. Wang, Tong Wu, Saeed Mahloujifar, Prateek Mittal

**Abstract**: Adversaries can embed backdoors in deep learning models by introducing backdoor poison samples into training datasets. In this work, we investigate how to detect such poison samples to mitigate the threat of backdoor attacks. First, we uncover a post-hoc workflow underlying most prior work, where defenders passively allow the attack to proceed and then leverage the characteristics of the post-attacked model to uncover poison samples. We reveal that this workflow does not fully exploit defenders' capabilities, and defense pipelines built on it are prone to failure or performance degradation in many scenarios. Second, we suggest a paradigm shift by promoting a proactive mindset in which defenders engage proactively with the entire model training and poison detection pipeline, directly enforcing and magnifying distinctive characteristics of the post-attacked model to facilitate poison detection. Based on this, we formulate a unified framework and provide practical insights on designing detection pipelines that are more robust and generalizable. Third, we introduce the technique of Confusion Training (CT) as a concrete instantiation of our framework. CT applies an additional poisoning attack to the already poisoned dataset, actively decoupling benign correlation while exposing backdoor patterns to detection. Empirical evaluations on 4 datasets and 14 types of attacks validate the superiority of CT over 11 baseline defenses.



## **28. Computing a Best Response against a Maximum Disruption Attack**

cs.GT

35 pages, 7 figures

**SubmitDate**: 2023-02-10    [abs](http://arxiv.org/abs/2302.05348v1) [paper-pdf](http://arxiv.org/pdf/2302.05348v1)

**Authors**: Carme Àlvarez, Arnau Messegué

**Abstract**: Inspired by scenarios where the strategic network design and defense or immunisation are of the central importance, Goyal et al. [3] defined a new Network Formation Game with Attack and Immunisation. The authors showed that despite the presence of attacks, the game has high social welfare properties and even though the equilibrium networks can contain cycles, the number of edges is strongly bounded. Subsequently, Friedrich et al. [10] provided a polynomial time algorithm for computing a best response strategy for the maximum carnage adversary which tries to kill as many nodes as possible, and for the random attack adversary, but they left open the problem for the case of maximum disruption adversary. This adversary attacks the vulnerable region that minimises the post-attack social welfare. In this paper we address our efforts to this question. We can show that computing a best response strategy given a player u and the strategies of all players but u, is polynomial time solvable when the initial network resulting from the given strategies is connected. Our algorithm is based on a dynamic programming and has some reminiscence to the knapsack-problem, although is considerably more complex and involved.



## **29. Step by Step Loss Goes Very Far: Multi-Step Quantization for Adversarial Text Attacks**

cs.CL

**SubmitDate**: 2023-02-10    [abs](http://arxiv.org/abs/2302.05120v1) [paper-pdf](http://arxiv.org/pdf/2302.05120v1)

**Authors**: Piotr Gaiński, Klaudia Bałazy

**Abstract**: We propose a novel gradient-based attack against transformer-based language models that searches for an adversarial example in a continuous space of token probabilities. Our algorithm mitigates the gap between adversarial loss for continuous and discrete text representations by performing multi-step quantization in a quantization-compensation loop. Experiments show that our method significantly outperforms other approaches on various natural language processing (NLP) tasks.



## **30. Making Substitute Models More Bayesian Can Enhance Transferability of Adversarial Examples**

cs.LG

Accepted by ICLR 2023

**SubmitDate**: 2023-02-10    [abs](http://arxiv.org/abs/2302.05086v1) [paper-pdf](http://arxiv.org/pdf/2302.05086v1)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: The transferability of adversarial examples across deep neural networks (DNNs) is the crux of many black-box attacks. Many prior efforts have been devoted to improving the transferability via increasing the diversity in inputs of some substitute models. In this paper, by contrast, we opt for the diversity in substitute models and advocate to attack a Bayesian model for achieving desirable transferability. Deriving from the Bayesian formulation, we develop a principled strategy for possible finetuning, which can be combined with many off-the-shelf Gaussian posterior approximations over DNN parameters. Extensive experiments have been conducted to verify the effectiveness of our method, on common benchmark datasets, and the results demonstrate that our method outperforms recent state-of-the-arts by large margins (roughly 19% absolute increase in average attack success rate on ImageNet), and, by combining with these recent methods, further performance gain can be obtained. Our code: https://github.com/qizhangli/MoreBayesian-attack.



## **31. Increasing-Margin Adversarial (IMA) Training to Improve Adversarial Robustness of Neural Networks**

cs.CV

26 pages

**SubmitDate**: 2023-02-10    [abs](http://arxiv.org/abs/2005.09147v10) [paper-pdf](http://arxiv.org/pdf/2005.09147v10)

**Authors**: Linhai Ma, Liang Liang

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial noises. Adversarial training is a general and effective strategy to improve DNN robustness (i.e., accuracy on noisy data) against adversarial noises. However, DNN models trained by the current existing adversarial training methods may have much lower standard accuracy (i.e., accuracy on clean data), compared to the same models trained by the standard method on clean data, and this phenomenon is known as the trade-off between accuracy and robustness and is considered unavoidable. This issue prevents adversarial training from being used in many application domains, such as medical image analysis, as practitioners do not want to sacrifice standard accuracy too much in exchange for adversarial robustness. Our objective is to lift (i.e., alleviate or even avoid) this trade-off between standard accuracy and adversarial robustness for medical image classification and segmentation. We propose a novel adversarial training method, named Increasing-Margin Adversarial (IMA) Training, which is supported by an equilibrium state analysis about the optimality of adversarial training samples. Our method aims to preserve accuracy while improving robustness by generating optimal adversarial training samples. We evaluate our method and the other eight representative methods on six publicly available image datasets corrupted by noises generated by AutoAttack and white-noise attack. Our method achieves the highest adversarial robustness for image classification and segmentation with the smallest reduction in accuracy on clean data. For one of the applications, our method improves both accuracy and robustness. Our study has demonstrated that our method can lift the trade-off between standard accuracy and adversarial robustness for the image classification and segmentation applications.



## **32. RAPTOR: Advanced Persistent Threat Detection in Industrial IoT via Attack Stage Correlation**

cs.CR

To be submitted to journal

**SubmitDate**: 2023-02-09    [abs](http://arxiv.org/abs/2301.11524v2) [paper-pdf](http://arxiv.org/pdf/2301.11524v2)

**Authors**: Ayush Kumar, Vrizlynn L. L. Thing

**Abstract**: IIoT (Industrial Internet-of-Things) systems are getting more prone to attacks by APT (Advanced Persistent Threat) adversaries. Past APT attacks on IIoT systems such as the 2016 Ukrainian power grid attack which cut off the capital Kyiv off power for an hour and the 2017 Saudi petrochemical plant attack which almost shut down the plant's safety controllers have shown that APT campaigns can disrupt industrial processes, shut down critical systems and endanger human lives. In this work, we propose RAPTOR, a system to detect APT campaigns in IIoT environments. RAPTOR detects and correlates various APT attack stages (adapted to IIoT) using multiple data sources. Subsequently, it constructs a high-level APT campaign graph which can be used by cybersecurity analysts towards attack analysis and mitigation. A performance evaluation of RAPTOR's APT stage detection stages shows high precision and low false positive/negative rates. We also show that RAPTOR is able to construct the APT campaign graph for APT attacks (modelled after real-world attacks on ICS/OT infrastructure) executed on our IIoT testbed.



## **33. Testing robustness of predictions of trained classifiers against naturally occurring perturbations**

cs.LG

25 pages, 7 figures

**SubmitDate**: 2023-02-09    [abs](http://arxiv.org/abs/2204.10046v2) [paper-pdf](http://arxiv.org/pdf/2204.10046v2)

**Authors**: Sebastian Scher, Andreas Trügler

**Abstract**: Correctly quantifying the robustness of machine learning models is a central aspect in judging their suitability for specific tasks, and ultimately, for generating trust in them. We address the problem of finding the robustness of individual predictions. We show both theoretically and with empirical examples that a method based on counterfactuals that was previously proposed for this is insufficient, as it is not a valid metric for determining the robustness against perturbations that occur ``naturally'', outside specific adversarial attack scenarios. We propose a flexible approach that models possible perturbations in input data individually for each application. This is then combined with a probabilistic approach that computes the likelihood that a ``real-world'' perturbation will change a prediction, thus giving quantitative information of the robustness of individual predictions of the trained machine learning model. The method does not require access to the internals of the classifier and thus in principle works for any black-box model. It is, however, based on Monte-Carlo sampling and thus only suited for input spaces with small dimensions. We illustrate our approach on the Iris and the Ionosphere datasets, on an application predicting fog at an airport, and on analytically solvable cases.



## **34. Tracking Fringe and Coordinated Activity on Twitter Leading Up To the US Capitol Attack**

cs.SI

11 pages (including references), 8 figures, 1 table. Submitted to The  17th International AAAI Conference on Web and Social Media

**SubmitDate**: 2023-02-09    [abs](http://arxiv.org/abs/2302.04450v1) [paper-pdf](http://arxiv.org/pdf/2302.04450v1)

**Authors**: Vishnuprasad Padinjaredath Suresh, Gianluca Nogara, Felipe Cardoso, Stefano Cresci, Silvia Giordano, Luca Luceri

**Abstract**: The aftermath of the 2020 US Presidential Election witnessed an unprecedented attack on the democratic values of the country through the violent insurrection at Capitol Hill on January 6th, 2021. The attack was fueled by the proliferation of conspiracy theories and misleading claims about the integrity of the election pushed by political elites and fringe communities on social media. In this study, we explore the evolution of fringe content and conspiracy theories on Twitter in the seven months leading up to the Capitol attack. We examine the suspicious coordinated activity carried out by users sharing fringe content, finding evidence of common adversarial manipulation techniques ranging from targeted amplification to manufactured consensus. Further, we map out the temporal evolution of, and the relationship between, fringe and conspiracy theories, which eventually coalesced into the rhetoric of a stolen election, with the hashtag #stopthesteal, alongside QAnon-related narratives. Our findings further highlight how social media platforms offer fertile ground for the widespread proliferation of conspiracies during major societal events, which can potentially lead to offline coordinated actions and organized violence.



## **35. Leveraging the Verifier's Dilemma to Double Spend in Bitcoin**

cs.CR

**SubmitDate**: 2023-02-09    [abs](http://arxiv.org/abs/2210.14072v2) [paper-pdf](http://arxiv.org/pdf/2210.14072v2)

**Authors**: Tong Cao, Jérémie Decouchant, Jiangshan Yu

**Abstract**: We describe and analyze perishing mining, a novel block-withholding mining strategy that lures profit-driven miners away from doing useful work on the public chain by releasing block headers from a privately maintained chain. We then introduce the dual private chain (DPC) attack, where an adversary that aims at double spending increases its success rate by intermittently dedicating part of its hash power to perishing mining. We detail the DPC attack's Markov decision process, evaluate its double spending success rate using Monte Carlo simulations. We show that the DPC attack lowers Bitcoin's security bound in the presence of profit-driven miners that do not wait to validate the transactions of a block before mining on it.



## **36. Exploiting Certified Defences to Attack Randomised Smoothing**

cs.LG

15 pages, 7 figures

**SubmitDate**: 2023-02-09    [abs](http://arxiv.org/abs/2302.04379v1) [paper-pdf](http://arxiv.org/pdf/2302.04379v1)

**Authors**: Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: In guaranteeing that no adversarial examples exist within a bounded region, certification mechanisms play an important role in neural network robustness. Concerningly, this work demonstrates that the certification mechanisms themselves introduce a new, heretofore undiscovered attack surface, that can be exploited by attackers to construct smaller adversarial perturbations. While these attacks exist outside the certification region in no way invalidate certifications, minimising a perturbation's norm significantly increases the level of difficulty associated with attack detection. In comparison to baseline attacks, our new framework yields smaller perturbations more than twice as frequently as any other approach, resulting in an up to $34 \%$ reduction in the median perturbation norm. That this approach also requires $90 \%$ less computational time than approaches like PGD. That these reductions are possible suggests that exploiting this new attack vector would allow attackers to more frequently construct hard to detect adversarial attacks, by exploiting the very systems designed to defend deployed models.



## **37. A Comprehensive Test Pattern Generation Approach Exploiting SAT Attack for Logic Locking**

cs.CR

12 pages, 7 figures, 5 tables

**SubmitDate**: 2023-02-08    [abs](http://arxiv.org/abs/2204.11307v4) [paper-pdf](http://arxiv.org/pdf/2204.11307v4)

**Authors**: Yadi Zhong, Ujjwal Guin

**Abstract**: The need for reducing manufacturing defect escape in today's safety-critical applications requires increased fault coverage. However, generating a test set using commercial automatic test pattern generation (ATPG) tools that lead to zero-defect escape is still an open problem. It is challenging to detect all stuck-at faults to reach 100% fault coverage. In parallel, the hardware security community has been actively involved in developing solutions for logic locking to prevent IP piracy. Locks (e.g., XOR gates) are inserted in different locations of the netlist so that an adversary cannot determine the secret key. Unfortunately, the Boolean satisfiability (SAT) based attack, introduced in [1], can break different logic locking schemes in minutes. In this paper, we propose a novel test pattern generation approach using the powerful SAT attack on logic locking. A stuck-at fault is modeled as a locked gate with a secret key. Our modeling of stuck-at faults preserves the property of fault activation and propagation. We show that the input pattern that determines the key is a test for the stuck-at fault. We propose two different approaches for test pattern generation. First, a single stuck-at fault is targeted, and a corresponding locked circuit with one key bit is created. This approach generates one test pattern per fault. Second, we consider a group of faults and convert the circuit to its locked version with multiple key bits. The inputs obtained from the SAT tool are the test set for detecting this group of faults. Our approach is able to find test patterns for hard-to-detect faults that were previously failed in commercial ATPG tools. The proposed test pattern generation approach can efficiently detect redundant faults present in a circuit. We demonstrate the effectiveness of the approach on ITC'99 benchmarks. The results show that we can achieve a perfect fault coverage reaching 100%.



## **38. WAT: Improve the Worst-class Robustness in Adversarial Training**

cs.LG

Accepted to AAAI 2023

**SubmitDate**: 2023-02-08    [abs](http://arxiv.org/abs/2302.04025v1) [paper-pdf](http://arxiv.org/pdf/2302.04025v1)

**Authors**: Boqi Li, Weiwei Liu

**Abstract**: Deep Neural Networks (DNN) have been shown to be vulnerable to adversarial examples. Adversarial training (AT) is a popular and effective strategy to defend against adversarial attacks. Recent works (Benz et al., 2020; Xu et al., 2021; Tian et al., 2021) have shown that a robust model well-trained by AT exhibits a remarkable robustness disparity among classes, and propose various methods to obtain consistent robust accuracy across classes. Unfortunately, these methods sacrifice a good deal of the average robust accuracy. Accordingly, this paper proposes a novel framework of worst-class adversarial training and leverages no-regret dynamics to solve this problem. Our goal is to obtain a classifier with great performance on worst-class and sacrifice just a little average robust accuracy at the same time. We then rigorously analyze the theoretical properties of our proposed algorithm, and the generalization error bound in terms of the worst-class robust risk. Furthermore, we propose a measurement to evaluate the proposed method in terms of both the average and worst-class accuracies. Experiments on various datasets and networks show that our proposed method outperforms the state-of-the-art approaches.



## **39. Adversarial Training of Self-supervised Monocular Depth Estimation against Physical-World Attacks**

cs.CV

Accepted at ICLR2023 (Spotlight), add code link

**SubmitDate**: 2023-02-08    [abs](http://arxiv.org/abs/2301.13487v2) [paper-pdf](http://arxiv.org/pdf/2301.13487v2)

**Authors**: Zhiyuan Cheng, James Liang, Guanhong Tao, Dongfang Liu, Xiangyu Zhang

**Abstract**: Monocular Depth Estimation (MDE) is a critical component in applications such as autonomous driving. There are various attacks against MDE networks. These attacks, especially the physical ones, pose a great threat to the security of such systems. Traditional adversarial training method requires ground-truth labels hence cannot be directly applied to self-supervised MDE that does not have ground-truth depth. Some self-supervised model hardening techniques (e.g., contrastive learning) ignore the domain knowledge of MDE and can hardly achieve optimal performance. In this work, we propose a novel adversarial training method for self-supervised MDE models based on view synthesis without using ground-truth depth. We improve adversarial robustness against physical-world attacks using L0-norm-bounded perturbation in training. We compare our method with supervised learning based and contrastive learning based methods that are tailored for MDE. Results on two representative MDE networks show that we achieve better robustness against various adversarial attacks with nearly no benign performance degradation.



## **40. Efficient Adversarial Contrastive Learning via Robustness-Aware Coreset Selection**

cs.LG

**SubmitDate**: 2023-02-08    [abs](http://arxiv.org/abs/2302.03857v1) [paper-pdf](http://arxiv.org/pdf/2302.03857v1)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL) does not require expensive data annotations but outputs a robust representation that withstands adversarial attacks and also generalizes to a wide range of downstream tasks. However, ACL needs tremendous running time to generate the adversarial variants of all training data, which limits its scalability to large datasets. To speed up ACL, this paper proposes a robustness-aware coreset selection (RCS) method. RCS does not require label information and searches for an informative subset that minimizes a representational divergence, which is the distance of the representation between natural data and their virtual adversarial variants. The vanilla solution of RCS via traversing all possible subsets is computationally prohibitive. Therefore, we theoretically transform RCS into a surrogate problem of submodular maximization, of which the greedy search is an efficient solution with an optimality guarantee for the original problem. Empirically, our comprehensive results corroborate that RCS can speed up ACL by a large margin without significantly hurting the robustness and standard transferability. Notably, to the best of our knowledge, we are the first to conduct ACL efficiently on the large-scale ImageNet-1K dataset to obtain an effective robust representation via RCS.



## **41. Temporal Robustness against Data Poisoning**

cs.LG

11 pages, 7 figures

**SubmitDate**: 2023-02-07    [abs](http://arxiv.org/abs/2302.03684v1) [paper-pdf](http://arxiv.org/pdf/2302.03684v1)

**Authors**: Wenxiao Wang, Soheil Feizi

**Abstract**: Data poisoning considers cases when an adversary maliciously inserts and removes training data to manipulate the behavior of machine learning algorithms. Traditional threat models of data poisoning center around a single metric, the number of poisoned samples. In consequence, existing defenses are essentially vulnerable in practice when poisoning more samples remains a feasible option for attackers. To address this issue, we leverage timestamps denoting the birth dates of data, which are often available but neglected in the past. Benefiting from these timestamps, we propose a temporal threat model of data poisoning and derive two novel metrics, earliness and duration, which respectively measure how long an attack started in advance and how long an attack lasted. With these metrics, we define the notions of temporal robustness against data poisoning, providing a meaningful sense of protection even with unbounded amounts of poisoned samples. We present a benchmark with an evaluation protocol simulating continuous data collection and periodic deployments of updated models, thus enabling empirical evaluation of temporal robustness. Lastly, we develop and also empirically verify a baseline defense, namely temporal aggregation, offering provable temporal robustness and highlighting the potential of our temporal modeling of data poisoning.



## **42. Universal Soldier: Using Universal Adversarial Perturbations for Detecting Backdoor Attacks**

cs.LG

**SubmitDate**: 2023-02-07    [abs](http://arxiv.org/abs/2302.00747v2) [paper-pdf](http://arxiv.org/pdf/2302.00747v2)

**Authors**: Xiaoyun Xu, Oguzhan Ersoy, Stjepan Picek

**Abstract**: Deep learning models achieve excellent performance in numerous machine learning tasks. Yet, they suffer from security-related issues such as adversarial examples and poisoning (backdoor) attacks. A deep learning model may be poisoned by training with backdoored data or by modifying inner network parameters. Then, a backdoored model performs as expected when receiving a clean input, but it misclassifies when receiving a backdoored input stamped with a pre-designed pattern called "trigger". Unfortunately, it is difficult to distinguish between clean and backdoored models without prior knowledge of the trigger. This paper proposes a backdoor detection method by utilizing a special type of adversarial attack, universal adversarial perturbation (UAP), and its similarities with a backdoor trigger. We observe an intuitive phenomenon: UAPs generated from backdoored models need fewer perturbations to mislead the model than UAPs from clean models. UAPs of backdoored models tend to exploit the shortcut from all classes to the target class, built by the backdoor trigger. We propose a novel method called Universal Soldier for Backdoor detection (USB) and reverse engineering potential backdoor triggers via UAPs. Experiments on 345 models trained on several datasets show that USB effectively detects the injected backdoor and provides comparable or better results than state-of-the-art methods.



## **43. Low-Latency Communication using Delay-Aware Relays Against Reactive Adversaries**

cs.IT

30 pages

**SubmitDate**: 2023-02-07    [abs](http://arxiv.org/abs/2302.03335v1) [paper-pdf](http://arxiv.org/pdf/2302.03335v1)

**Authors**: Vivek Chaudhary, J. Harshan

**Abstract**: This work addresses a reactive jamming attack on the low-latency messages of a victim, wherein the jammer deploys countermeasure detection mechanisms to change its strategy. We highlight that the existing schemes against reactive jammers use relays with instantaneous full-duplex (FD) radios to evade the attack. However, due to the limitation of the radio architecture of the FD helper, instantaneous forwarding may not be possible in practice, thereby leading to increased decoding complexity at the destination and a high detection probability at the adversary. Pointing at this drawback, we propose a delay-aware cooperative framework wherein the victim seeks assistance from a delay-aware FD helper to forward its messages to the destination within the latency constraints. In particular, we first model the processing delay at the helper based on its hardware architecture, and then propose two low-complexity mitigation schemes, wherein the victim and the helper share their uplink frequencies using appropriate energy-splitting factors. For both the schemes, we solve the optimization problems of computing the near-optimal energy-splitting factors that minimize the joint error rates at the destination. Finally, through analytical and simulation results, we show that the proposed schemes facilitate the victim in evading the jamming attack whilst deceiving the reactive adversary.



## **44. Attacking Cooperative Multi-Agent Reinforcement Learning by Adversarial Minority Influence**

cs.LG

**SubmitDate**: 2023-02-07    [abs](http://arxiv.org/abs/2302.03322v1) [paper-pdf](http://arxiv.org/pdf/2302.03322v1)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Pu Feng, Xin Yu, Jiakai Wang, Aishan Liu, Wenjun Wu, Xianglong Liu

**Abstract**: Cooperative multi-agent reinforcement learning (c-MARL) offers a general paradigm for a group of agents to achieve a shared goal by taking individual decisions, yet is found to be vulnerable to adversarial attacks. Though harmful, adversarial attacks also play a critical role in evaluating the robustness and finding blind spots of c-MARL algorithms. However, existing attacks are not sufficiently strong and practical, which is mainly due to the ignorance of complex influence between agents and cooperative nature of victims in c-MARL.   In this paper, we propose adversarial minority influence (AMI), the first practical attack against c-MARL by introducing an adversarial agent. AMI addresses the aforementioned problems by unilaterally influencing other cooperative victims to a targeted worst-case cooperation. Technically, to maximally deviate victim policy under complex agent-wise influence, our unilateral attack characterize and maximize the influence from adversary to victims. This is done by adapting a unilateral agent-wise relation metric derived from mutual information, which filters out the detrimental influence from victims to adversary. To fool victims into a jointly worst-case failure, our targeted attack influence victims to a long-term, cooperatively worst case by distracting each victim to a specific target. Such target is learned by a reinforcement learning agent in a trial-and-error process. Extensive experiments in simulation environments, including discrete control (SMAC), continuous control (MAMujoco) and real-world robot swarm control demonstrate the superiority of our AMI approach. Our codes are available in https://anonymous.4open.science/r/AMI.



## **45. Universal Coding for Shannon Ciphers under Side-Channel Attacks**

cs.IT

6 pages, 3 figures. previous version has some mistake on the problem  set up and the order of the authors. We correct those mistakes in this  version. The problem set up in this paper is the same as the one proposed by  Santoso and Oohama (Entropy 2019) but is different from the one proposed by  Oohama and Santoso (ISIT 2022, arXivarXiv:2201.11670). arXiv admin note:  substantial text overlap with arXiv:1801.02563, arXiv:2201.11670,  arXiv:1901.05940

**SubmitDate**: 2023-02-07    [abs](http://arxiv.org/abs/2302.01314v2) [paper-pdf](http://arxiv.org/pdf/2302.01314v2)

**Authors**: Bagus Santoso, Yasutada Oohama

**Abstract**: We study the universal coding under side-channel attacks posed and investigated by Santoso and Oohama (2019). They proposed a theoretical security model for Shannon cipher system under side-channel attacks, where the adversary is not only allowed to collect ciphertexts by eavesdropping the public communication channel, but is also allowed to collect the physical information leaked by the devices where the cipher system is implemented on such as running time, power consumption, electromagnetic radiation, etc. For any distributions of the plain text, any noisy channels through which the adversary observe the corrupted version of the key, and any measurement device used for collecting the physical information, we can derive an achievable rate region for reliability and security such that if we compress the ciphertext using an affine encoder with rate within the achievable rate region, then: (1) anyone with secret key will be able to decrypt and decode the ciphertext correctly, but (2) any adversary who obtains the ciphertext and also the   side physical information will not be able to obtain any information about the hidden source as long as the leaked physical information is encoded with a rate within the rate region.



## **46. Towards Out-of-Distribution Adversarial Robustness**

cs.LG

**SubmitDate**: 2023-02-07    [abs](http://arxiv.org/abs/2210.03150v3) [paper-pdf](http://arxiv.org/pdf/2210.03150v3)

**Authors**: Adam Ibrahim, Charles Guille-Escuret, Ioannis Mitliagkas, Irina Rish, David Krueger, Pouya Bashivan

**Abstract**: Adversarial robustness continues to be a major challenge for deep learning. A core issue is that robustness to one type of attack often fails to transfer to other attacks. While prior work establishes a theoretical trade-off in robustness against different $L_p$ norms, we show that there is potential for improvement against many commonly used attacks by adopting a domain generalisation approach. Concretely, we treat each type of attack as a domain, and apply the Risk Extrapolation method (REx), which promotes similar levels of robustness against all training attacks. Compared to existing methods, we obtain similar or superior worst-case adversarial robustness on attacks seen during training. Moreover, we achieve superior performance on families or tunings of attacks only encountered at test time. On ensembles of attacks, our approach improves the accuracy from 3.4% the best existing baseline to 25.9% on MNIST, and from 16.9% to 23.5% on CIFAR10.



## **47. Less is More: Understanding Word-level Textual Adversarial Attack via n-gram Frequency Descend**

cs.CL

8 pages, 4 figures. In progress

**SubmitDate**: 2023-02-07    [abs](http://arxiv.org/abs/2302.02568v2) [paper-pdf](http://arxiv.org/pdf/2302.02568v2)

**Authors**: Ning Lu, Shengcai Liu, Zhirui Zhang, Qi Wang, Haifeng Liu, Ke Tang

**Abstract**: Word-level textual adversarial attacks have achieved striking performance in fooling natural language processing models. However, the fundamental questions of why these attacks are effective, and the intrinsic properties of the adversarial examples (AEs), are still not well understood. This work attempts to interpret textual attacks through the lens of $n$-gram frequency. Specifically, it is revealed that existing word-level attacks exhibit a strong tendency toward generation of examples with $n$-gram frequency descend ($n$-FD). Intuitively, this finding suggests a natural way to improve model robustness by training the model on the $n$-FD examples. To verify this idea, we devise a model-agnostic and gradient-free AE generation approach that relies solely on the $n$-gram frequency information, and further integrate it into the recently proposed convex hull framework for adversarial training. Surprisingly, the resultant method performs quite similarly to the original gradient-based method in terms of model robustness. These findings provide a human-understandable perspective for interpreting word-level textual adversarial attacks, and a new direction to improve model robustness.



## **48. Projective Ranking-based GNN Evasion Attacks**

cs.LG

Accepted by IEEE Transactions on Knowledge and Data Engineering

**SubmitDate**: 2023-02-07    [abs](http://arxiv.org/abs/2202.12993v3) [paper-pdf](http://arxiv.org/pdf/2202.12993v3)

**Authors**: He Zhang, Xingliang Yuan, Chuan Zhou, Shirui Pan

**Abstract**: Graph neural networks (GNNs) offer promising learning methods for graph-related tasks. However, GNNs are at risk of adversarial attacks. Two primary limitations of the current evasion attack methods are highlighted: (1) The current GradArgmax ignores the "long-term" benefit of the perturbation. It is faced with zero-gradient and invalid benefit estimates in certain situations. (2) In the reinforcement learning-based attack methods, the learned attack strategies might not be transferable when the attack budget changes. To this end, we first formulate the perturbation space and propose an evaluation framework and the projective ranking method. We aim to learn a powerful attack strategy then adapt it as little as possible to generate adversarial samples under dynamic budget settings. In our method, based on mutual information, we rank and assess the attack benefits of each perturbation for an effective attack strategy. By projecting the strategy, our method dramatically minimizes the cost of learning a new attack strategy when the attack budget changes. In the comparative assessment with GradArgmax and RL-S2V, the results show our method owns high attack performance and effective transferability. The visualization of our method also reveals various attack patterns in the generation of adversarial samples.



## **49. Membership Inference Attacks against Diffusion Models**

cs.CR

**SubmitDate**: 2023-02-07    [abs](http://arxiv.org/abs/2302.03262v1) [paper-pdf](http://arxiv.org/pdf/2302.03262v1)

**Authors**: Tomoya Matsumoto, Takayuki Miura, Naoto Yanai

**Abstract**: Diffusion models have attracted attention in recent years as innovative generative models. In this paper, we investigate whether a diffusion model is resistant to a membership inference attack, which evaluates the privacy leakage of a machine learning model. We primarily discuss the diffusion model from the standpoints of comparison with a generative adversarial network (GAN) as conventional models and hyperparameters unique to the diffusion model, i.e., time steps, sampling steps, and sampling variances. We conduct extensive experiments with DDIM as a diffusion model and DCGAN as a GAN on the CelebA and CIFAR-10 datasets in both white-box and black-box settings and then confirm if the diffusion model is comparably resistant to a membership inference attack as GAN. Next, we demonstrate that the impact of time steps is significant and intermediate steps in a noise schedule are the most vulnerable to the attack. We also found two key insights through further analysis. First, we identify that DDIM is vulnerable to the attack for small sample sizes instead of achieving a lower FID. Second, sampling steps in hyperparameters are important for resistance to the attack, whereas the impact of sampling variances is quite limited.



## **50. SCALE-UP: An Efficient Black-box Input-level Backdoor Detection via Analyzing Scaled Prediction Consistency**

cs.CR

**SubmitDate**: 2023-02-07    [abs](http://arxiv.org/abs/2302.03251v1) [paper-pdf](http://arxiv.org/pdf/2302.03251v1)

**Authors**: Junfeng Guo, Yiming Li, Xun Chen, Hanqing Guo, Lichao Sun, Cong Liu

**Abstract**: Deep neural networks (DNNs) are vulnerable to backdoor attacks, where adversaries embed a hidden backdoor trigger during the training process for malicious prediction manipulation. These attacks pose great threats to the applications of DNNs under the real-world machine learning as a service (MLaaS) setting, where the deployed model is fully black-box while the users can only query and obtain its predictions. Currently, there are many existing defenses to reduce backdoor threats. However, almost all of them cannot be adopted in MLaaS scenarios since they require getting access to or even modifying the suspicious models. In this paper, we propose a simple yet effective black-box input-level backdoor detection, called SCALE-UP, which requires only the predicted labels to alleviate this problem. Specifically, we identify and filter malicious testing samples by analyzing their prediction consistency during the pixel-wise amplification process. Our defense is motivated by an intriguing observation (dubbed scaled prediction consistency) that the predictions of poisoned samples are significantly more consistent compared to those of benign ones when amplifying all pixel values. Besides, we also provide theoretical foundations to explain this phenomenon. Extensive experiments are conducted on benchmark datasets, verifying the effectiveness and efficiency of our defense and its resistance to potential adaptive attacks. Our codes are available at https://github.com/JunfengGo/SCALE-UP.



