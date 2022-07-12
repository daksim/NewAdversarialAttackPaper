# Latest Adversarial Attack Papers
**update at 2022-07-13 06:31:27**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Risk assessment and optimal allocation of security measures under stealthy false data injection attacks**

eess.SY

Accepted for publication at 6th IEEE Conference on Control Technology  and Applications (CCTA). arXiv admin note: substantial text overlap with  arXiv:2106.07071

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.04860v1)

**Authors**: Sribalaji C. Anand, André M. H. Teixeira, Anders Ahlén

**Abstracts**: This paper firstly addresses the problem of risk assessment under false data injection attacks on uncertain control systems. We consider an adversary with complete system knowledge, injecting stealthy false data into an uncertain control system. We then use the Value-at-Risk to characterize the risk associated with the attack impact caused by the adversary. The worst-case attack impact is characterized by the recently proposed output-to-output gain. We observe that the risk assessment problem corresponds to an infinite non-convex robust optimization problem. To this end, we use dissipative system theory and the scenario approach to approximate the risk-assessment problem into a convex problem and also provide probabilistic certificates on approximation. Secondly, we consider the problem of security measure allocation. We consider an operator with a constraint on the security budget. Under this constraint, we propose an algorithm to optimally allocate the security measures using the calculated risk such that the resulting Value-at-risk is minimized. Finally, we illustrate the results through a numerical example. The numerical example also illustrates that the security allocation using the Value-at-risk, and the impact on the nominal system may have different outcomes: thereby depicting the benefit of using risk metrics.



## **2. Statistical Detection of Adversarial examples in Blockchain-based Federated Forest In-vehicle Network Intrusion Detection Systems**

cs.CR

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.04843v1)

**Authors**: Ibrahim Aliyu, Selinde van Engelenburg, Muhammed Bashir Muazu, Jinsul Kim, Chang Gyoon Lim

**Abstracts**: The internet-of-Vehicle (IoV) can facilitate seamless connectivity between connected vehicles (CV), autonomous vehicles (AV), and other IoV entities. Intrusion Detection Systems (IDSs) for IoV networks can rely on machine learning (ML) to protect the in-vehicle network from cyber-attacks. Blockchain-based Federated Forests (BFFs) could be used to train ML models based on data from IoV entities while protecting the confidentiality of the data and reducing the risks of tampering with the data. However, ML models created this way are still vulnerable to evasion, poisoning, and exploratory attacks using adversarial examples. This paper investigates the impact of various possible adversarial examples on the BFF-IDS. We proposed integrating a statistical detector to detect and extract unknown adversarial samples. By including the unknown detected samples into the dataset of the detector, we augment the BFF-IDS with an additional model to detect original known attacks and the new adversarial inputs. The statistical adversarial detector confidently detected adversarial examples at the sample size of 50 and 100 input samples. Furthermore, the augmented BFF-IDS (BFF-IDS(AUG)) successfully mitigates the adversarial examples with more than 96% accuracy. With this approach, the model will continue to be augmented in a sandbox whenever an adversarial sample is detected and subsequently adopt the BFF-IDS(AUG) as the active security model. Consequently, the proposed integration of the statistical adversarial detector and the subsequent augmentation of the BFF-IDS with detected adversarial samples provides a sustainable security framework against adversarial examples and other unknown attacks.



## **3. Physical Attack on Monocular Depth Estimation with Optimal Adversarial Patches**

cs.CV

ECCV2022

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.04718v1)

**Authors**: Zhiyuan Cheng, James Liang, Hongjun Choi, Guanhong Tao, Zhiwen Cao, Dongfang Liu, Xiangyu Zhang

**Abstracts**: Deep learning has substantially boosted the performance of Monocular Depth Estimation (MDE), a critical component in fully vision-based autonomous driving (AD) systems (e.g., Tesla and Toyota). In this work, we develop an attack against learning-based MDE. In particular, we use an optimization-based method to systematically generate stealthy physical-object-oriented adversarial patches to attack depth estimation. We balance the stealth and effectiveness of our attack with object-oriented adversarial design, sensitive region localization, and natural style camouflage. Using real-world driving scenarios, we evaluate our attack on concurrent MDE models and a representative downstream task for AD (i.e., 3D object detection). Experimental results show that our method can generate stealthy, effective, and robust adversarial patches for different target objects and models and achieves more than 6 meters mean depth estimation error and 93% attack success rate (ASR) in object detection with a patch of 1/9 of the vehicle's rear area. Field tests on three different driving routes with a real vehicle indicate that we cause over 6 meters mean depth estimation error and reduce the object detection rate from 90.70% to 5.16% in continuous video frames.



## **4. Universal Adversarial Examples in Remote Sensing: Methodology and Benchmark**

cs.CV

**SubmitDate**: 2022-07-10    [paper-pdf](http://arxiv.org/pdf/2202.07054v2)

**Authors**: Yonghao Xu, Pedram Ghamisi

**Abstracts**: Deep neural networks have achieved great success in many important remote sensing tasks. Nevertheless, their vulnerability to adversarial examples should not be neglected. In this study, we systematically analyze the universal adversarial examples in remote sensing data for the first time, without any knowledge from the victim model. Specifically, we propose a novel black-box adversarial attack method, namely Mixup-Attack, and its simple variant Mixcut-Attack, for remote sensing data. The key idea of the proposed methods is to find common vulnerabilities among different networks by attacking the features in the shallow layer of a given surrogate model. Despite their simplicity, the proposed methods can generate transferable adversarial examples that deceive most of the state-of-the-art deep neural networks in both scene classification and semantic segmentation tasks with high success rates. We further provide the generated universal adversarial examples in the dataset named UAE-RS, which is the first dataset that provides black-box adversarial samples in the remote sensing field. We hope UAE-RS may serve as a benchmark that helps researchers to design deep neural networks with strong resistance toward adversarial attacks in the remote sensing field. Codes and the UAE-RS dataset are available online (https://github.com/YonghaoXu/UAE-RS).



## **5. Visual explanation of black-box model: Similarity Difference and Uniqueness (SIDU) method**

cs.CV

**SubmitDate**: 2022-07-10    [paper-pdf](http://arxiv.org/pdf/2101.10710v2)

**Authors**: Satya M. Muddamsetty, Mohammad N. S. Jahromi, Andreea E. Ciontos, Laura M. Fenoy, Thomas B. Moeslund

**Abstracts**: Explainable Artificial Intelligence (XAI) has in recent years become a well-suited framework to generate human understandable explanations of "black-box" models. In this paper, a novel XAI visual explanation algorithm known as the Similarity Difference and Uniqueness (SIDU) method that can effectively localize entire object regions responsible for prediction is presented in full detail. The SIDU algorithm robustness and effectiveness is analyzed through various computational and human subject experiments. In particular, the SIDU algorithm is assessed using three different types of evaluations (Application, Human and Functionally-Grounded) to demonstrate its superior performance. The robustness of SIDU is further studied in the presence of adversarial attack on "black-box" models to better understand its performance. Our code is available at: https://github.com/satyamahesh84/SIDU_XAI_CODE.



## **6. Fooling Partial Dependence via Data Poisoning**

cs.LG

Accepted at ECML PKDD 2022

**SubmitDate**: 2022-07-10    [paper-pdf](http://arxiv.org/pdf/2105.12837v3)

**Authors**: Hubert Baniecki, Wojciech Kretowicz, Przemyslaw Biecek

**Abstracts**: Many methods have been developed to understand complex predictive models and high expectations are placed on post-hoc model explainability. It turns out that such explanations are not robust nor trustworthy, and they can be fooled. This paper presents techniques for attacking Partial Dependence (plots, profiles, PDP), which are among the most popular methods of explaining any predictive model trained on tabular data. We showcase that PD can be manipulated in an adversarial manner, which is alarming, especially in financial or medical applications where auditability became a must-have trait supporting black-box machine learning. The fooling is performed via poisoning the data to bend and shift explanations in the desired direction using genetic and gradient algorithms. We believe this to be the first work using a genetic algorithm for manipulating explanations, which is transferable as it generalizes both ways: in a model-agnostic and an explanation-agnostic manner.



## **7. Adversarial Framework with Certified Robustness for Time-Series Domain via Statistical Features**

cs.LG

Published at Journal of Artificial Intelligence Research

**SubmitDate**: 2022-07-09    [paper-pdf](http://arxiv.org/pdf/2207.04307v1)

**Authors**: Taha Belkhouja, Janardhan Rao Doppa

**Abstracts**: Time-series data arises in many real-world applications (e.g., mobile health) and deep neural networks (DNNs) have shown great success in solving them. Despite their success, little is known about their robustness to adversarial attacks. In this paper, we propose a novel adversarial framework referred to as Time-Series Attacks via STATistical Features (TSA-STAT)}. To address the unique challenges of time-series domain, TSA-STAT employs constraints on statistical features of the time-series data to construct adversarial examples. Optimized polynomial transformations are used to create attacks that are more effective (in terms of successfully fooling DNNs) than those based on additive perturbations. We also provide certified bounds on the norm of the statistical features for constructing adversarial examples. Our experiments on diverse real-world benchmark datasets show the effectiveness of TSA-STAT in fooling DNNs for time-series domain and in improving their robustness. The source code of TSA-STAT algorithms is available at https://github.com/tahabelkhouja/Time-Series-Attacks-via-STATistical-Features



## **8. Not all broken defenses are equal: The dead angles of adversarial accuracy**

cs.LG

**SubmitDate**: 2022-07-08    [paper-pdf](http://arxiv.org/pdf/2207.04129v1)

**Authors**: Raphael Olivier, Bhiksha Raj

**Abstracts**: Robustness to adversarial attack is typically evaluated with adversarial accuracy. This metric is however too coarse to properly capture all robustness properties of machine learning models. Many defenses, when evaluated against a strong attack, do not provide accuracy improvements while still contributing partially to adversarial robustness. Popular certification methods suffer from the same issue, as they provide a lower bound to accuracy. To capture finer robustness properties we propose a new metric for L2 robustness, adversarial angular sparsity, which partially answers the question "how many adversarial examples are there around an input". We demonstrate its usefulness by evaluating both "strong" and "weak" defenses. We show that some state-of-the-art defenses, delivering very similar accuracy, can have very different sparsity on the inputs that they are not robust on. We also show that some weak defenses actually decrease robustness, while others strengthen it in a measure that accuracy cannot capture. These differences are predictive of how useful such defenses can become when combined with adversarial training.



## **9. Neighbors From Hell: Voltage Attacks Against Deep Learning Accelerators on Multi-Tenant FPGAs**

cs.CR

Published in the 2020 proceedings of the International Conference of  Field-Programmable Technology (ICFPT)

**SubmitDate**: 2022-07-08    [paper-pdf](http://arxiv.org/pdf/2012.07242v2)

**Authors**: Andrew Boutros, Mathew Hall, Nicolas Papernot, Vaughn Betz

**Abstracts**: Field-programmable gate arrays (FPGAs) are becoming widely used accelerators for a myriad of datacenter applications due to their flexibility and energy efficiency. Among these applications, FPGAs have shown promising results in accelerating low-latency real-time deep learning (DL) inference, which is becoming an indispensable component of many end-user applications. With the emerging research direction towards virtualized cloud FPGAs that can be shared by multiple users, the security aspect of FPGA-based DL accelerators requires careful consideration. In this work, we evaluate the security of DL accelerators against voltage-based integrity attacks in a multitenant FPGA scenario. We first demonstrate the feasibility of such attacks on a state-of-the-art Stratix 10 card using different attacker circuits that are logically and physically isolated in a separate attacker role, and cannot be flagged as malicious circuits by conventional bitstream checkers. We show that aggressive clock gating, an effective power-saving technique, can also be a potential security threat in modern FPGAs. Then, we carry out the attack on a DL accelerator running ImageNet classification in the victim role to evaluate the inherent resilience of DL models against timing faults induced by the adversary. We find that even when using the strongest attacker circuit, the prediction accuracy of the DL accelerator is not compromised when running at its safe operating frequency. Furthermore, we can achieve 1.18-1.31x higher inference performance by over-clocking the DL accelerator without affecting its prediction accuracy.



## **10. Defense Against Multi-target Trojan Attacks**

cs.CV

**SubmitDate**: 2022-07-08    [paper-pdf](http://arxiv.org/pdf/2207.03895v1)

**Authors**: Haripriya Harikumar, Santu Rana, Kien Do, Sunil Gupta, Wei Zong, Willy Susilo, Svetha Venkastesh

**Abstracts**: Adversarial attacks on deep learning-based models pose a significant threat to the current AI infrastructure. Among them, Trojan attacks are the hardest to defend against. In this paper, we first introduce a variation of the Badnet kind of attacks that introduces Trojan backdoors to multiple target classes and allows triggers to be placed anywhere in the image. The former makes it more potent and the latter makes it extremely easy to carry out the attack in the physical space. The state-of-the-art Trojan detection methods fail with this threat model. To defend against this attack, we first introduce a trigger reverse-engineering mechanism that uses multiple images to recover a variety of potential triggers. We then propose a detection mechanism by measuring the transferability of such recovered triggers. A Trojan trigger will have very high transferability i.e. they make other images also go to the same class. We study many practical advantages of our attack method and then demonstrate the detection performance using a variety of image datasets. The experimental results show the superior detection performance of our method over the state-of-the-arts.



## **11. Towards Effective and Robust Neural Trojan Defenses via Input Filtering**

cs.CR

Accepted to ECCV 2022

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2202.12154v4)

**Authors**: Kien Do, Haripriya Harikumar, Hung Le, Dung Nguyen, Truyen Tran, Santu Rana, Dang Nguyen, Willy Susilo, Svetha Venkatesh

**Abstracts**: Trojan attacks on deep neural networks are both dangerous and surreptitious. Over the past few years, Trojan attacks have advanced from using only a single input-agnostic trigger and targeting only one class to using multiple, input-specific triggers and targeting multiple classes. However, Trojan defenses have not caught up with this development. Most defense methods still make inadequate assumptions about Trojan triggers and target classes, thus, can be easily circumvented by modern Trojan attacks. To deal with this problem, we propose two novel "filtering" defenses called Variational Input Filtering (VIF) and Adversarial Input Filtering (AIF) which leverage lossy data compression and adversarial learning respectively to effectively purify potential Trojan triggers in the input at run time without making assumptions about the number of triggers/target classes or the input dependence property of triggers. In addition, we introduce a new defense mechanism called "Filtering-then-Contrasting" (FtC) which helps avoid the drop in classification accuracy on clean data caused by "filtering", and combine it with VIF/AIF to derive new defenses of this kind. Extensive experimental results and ablation studies show that our proposed defenses significantly outperform well-known baseline defenses in mitigating five advanced Trojan attacks including two recent state-of-the-art while being quite robust to small amounts of training data and large-norm triggers.



## **12. On the Relationship Between Adversarial Robustness and Decision Region in Deep Neural Network**

cs.LG

14 pages

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2207.03400v1)

**Authors**: Seongjin Park, Haedong Jeong, Giyoung Jeon, Jaesik Choi

**Abstracts**: In general, Deep Neural Networks (DNNs) are evaluated by the generalization performance measured on unseen data excluded from the training phase. Along with the development of DNNs, the generalization performance converges to the state-of-the-art and it becomes difficult to evaluate DNNs solely based on this metric. The robustness against adversarial attack has been used as an additional metric to evaluate DNNs by measuring their vulnerability. However, few studies have been performed to analyze the adversarial robustness in terms of the geometry in DNNs. In this work, we perform an empirical study to analyze the internal properties of DNNs that affect model robustness under adversarial attacks. In particular, we propose the novel concept of the Populated Region Set (PRS), where training samples are populated more frequently, to represent the internal properties of DNNs in a practical setting. From systematic experiments with the proposed concept, we provide empirical evidence to validate that a low PRS ratio has a strong relationship with the adversarial robustness of DNNs. We also devise PRS regularizer leveraging the characteristics of PRS to improve the adversarial robustness without adversarial training.



## **13. Federated Robustness Propagation: Sharing Robustness in Heterogeneous Federated Learning**

cs.LG

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2106.10196v2)

**Authors**: Junyuan Hong, Haotao Wang, Zhangyang Wang, Jiayu Zhou

**Abstracts**: Federated learning (FL) emerges as a popular distributed learning schema that learns a model from a set of participating users without sharing raw data. One major challenge of FL comes with heterogeneous users, who may have distributionally different (or non-iid) data and varying computation resources. As federated users would use the model for prediction, they often demand the trained model to be robust against malicious attackers at test time. Whereas adversarial training (AT) provides a sound solution for centralized learning, extending its usage for federated users has imposed significant challenges, as many users may have very limited training data and tight computational budgets, to afford the data-hungry and costly AT. In this paper, we study a novel FL strategy: propagating adversarial robustness from rich-resource users that can afford AT, to those with poor resources that cannot afford it, during federated learning. We show that existing FL techniques cannot be effectively integrated with the strategy to propagate robustness among non-iid users and propose an efficient propagation approach by the proper use of batch-normalization. We demonstrate the rationality and effectiveness of our method through extensive experiments. Especially, the proposed method is shown to grant federated models remarkable robustness even when only a small portion of users afford AT during learning. Source code will be released.



## **14. SYNFI: Pre-Silicon Fault Analysis of an Open-Source Secure Element**

cs.CR

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2205.04775v2)

**Authors**: Pascal Nasahl, Miguel Osorio, Pirmin Vogel, Michael Schaffner, Timothy Trippel, Dominic Rizzo, Stefan Mangard

**Abstracts**: Fault attacks are active, physical attacks that an adversary can leverage to alter the control-flow of embedded devices to gain access to sensitive information or bypass protection mechanisms. Due to the severity of these attacks, manufacturers deploy hardware-based fault defenses into security-critical systems, such as secure elements. The development of these countermeasures is a challenging task due to the complex interplay of circuit components and because contemporary design automation tools tend to optimize inserted structures away, thereby defeating their purpose. Hence, it is critical that such countermeasures are rigorously verified post-synthesis. As classical functional verification techniques fall short of assessing the effectiveness of countermeasures, developers have to resort to methods capable of injecting faults in a simulation testbench or into a physical chip. However, developing test sequences to inject faults in simulation is an error-prone task and performing fault attacks on a chip requires specialized equipment and is incredibly time-consuming. To that end, this paper introduces SYNFI, a formal pre-silicon fault verification framework that operates on synthesized netlists. SYNFI can be used to analyze the general effect of faults on the input-output relationship in a circuit and its fault countermeasures, and thus enables hardware designers to assess and verify the effectiveness of embedded countermeasures in a systematic and semi-automatic way. To demonstrate that SYNFI is capable of handling unmodified, industry-grade netlists synthesized with commercial and open tools, we analyze OpenTitan, the first open-source secure element. In our analysis, we identified critical security weaknesses in the unprotected AES block, developed targeted countermeasures, reassessed their security, and contributed these countermeasures back to the OpenTitan repository.



## **15. The StarCraft Multi-Agent Challenges+ : Learning of Multi-Stage Tasks and Environmental Factors without Precise Reward Functions**

cs.LG

ICML Workshop: AI for Agent Based Modeling 2022 Spotlight

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2207.02007v2)

**Authors**: Mingyu Kim, Jihwan Oh, Yongsik Lee, Joonkee Kim, Seonghwan Kim, Song Chong, Se-Young Yun

**Abstracts**: In this paper, we propose a novel benchmark called the StarCraft Multi-Agent Challenges+, where agents learn to perform multi-stage tasks and to use environmental factors without precise reward functions. The previous challenges (SMAC) recognized as a standard benchmark of Multi-Agent Reinforcement Learning are mainly concerned with ensuring that all agents cooperatively eliminate approaching adversaries only through fine manipulation with obvious reward functions. This challenge, on the other hand, is interested in the exploration capability of MARL algorithms to efficiently learn implicit multi-stage tasks and environmental factors as well as micro-control. This study covers both offensive and defensive scenarios. In the offensive scenarios, agents must learn to first find opponents and then eliminate them. The defensive scenarios require agents to use topographic features. For example, agents need to position themselves behind protective structures to make it harder for enemies to attack. We investigate MARL algorithms under SMAC+ and observe that recent approaches work well in similar settings to the previous challenges, but misbehave in offensive scenarios. Additionally, we observe that an enhanced exploration approach has a positive effect on performance but is not able to completely solve all scenarios. This study proposes new directions for future research.



## **16. The Weaknesses of Adversarial Camouflage in Overhead Imagery**

cs.CV

7 pages, 15 figures

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2207.02963v1)

**Authors**: Adam Van Etten

**Abstracts**: Machine learning is increasingly critical for analysis of the ever-growing corpora of overhead imagery. Advanced computer vision object detection techniques have demonstrated great success in identifying objects of interest such as ships, automobiles, and aircraft from satellite and drone imagery. Yet relying on computer vision opens up significant vulnerabilities, namely, the susceptibility of object detection algorithms to adversarial attacks. In this paper we explore the efficacy and drawbacks of adversarial camouflage in an overhead imagery context. While a number of recent papers have demonstrated the ability to reliably fool deep learning classifiers and object detectors with adversarial patches, most of this work has been performed on relatively uniform datasets and only a single class of objects. In this work we utilize the VisDrone dataset, which has a large range of perspectives and object sizes. We explore four different object classes: bus, car, truck, van. We build a library of 24 adversarial patches to disguise these objects, and introduce a patch translucency variable to our patches. The translucency (or alpha value) of the patches is highly correlated to their efficacy. Further, we show that while adversarial patches may fool object detectors, the presence of such patches is often easily uncovered, with patches on average 24% more detectable than the objects the patches were meant to hide. This raises the question of whether such patches truly constitute camouflage. Source code is available at https://github.com/IQTLabs/camolo.



## **17. DeepAdversaries: Examining the Robustness of Deep Learning Models for Galaxy Morphology Classification**

cs.LG

20 pages, 6 figures, 5 tables; accepted in MLST

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2112.14299v3)

**Authors**: Aleksandra Ćiprijanović, Diana Kafkes, Gregory Snyder, F. Javier Sánchez, Gabriel Nathan Perdue, Kevin Pedro, Brian Nord, Sandeep Madireddy, Stefan M. Wild

**Abstracts**: With increased adoption of supervised deep learning methods for processing and analysis of cosmological survey data, the assessment of data perturbation effects (that can naturally occur in the data processing and analysis pipelines) and the development of methods that increase model robustness are increasingly important. In the context of morphological classification of galaxies, we study the effects of perturbations in imaging data. In particular, we examine the consequences of using neural networks when training on baseline data and testing on perturbed data. We consider perturbations associated with two primary sources: 1) increased observational noise as represented by higher levels of Poisson noise and 2) data processing noise incurred by steps such as image compression or telescope errors as represented by one-pixel adversarial attacks. We also test the efficacy of domain adaptation techniques in mitigating the perturbation-driven errors. We use classification accuracy, latent space visualizations, and latent space distance to assess model robustness. Without domain adaptation, we find that processing pixel-level errors easily flip the classification into an incorrect class and that higher observational noise makes the model trained on low-noise data unable to classify galaxy morphologies. On the other hand, we show that training with domain adaptation improves model robustness and mitigates the effects of these perturbations, improving the classification accuracy by 23% on data with higher observational noise. Domain adaptation also increases by a factor of ~2.3 the latent space distance between the baseline and the incorrectly classified one-pixel perturbed image, making the model more robust to inadvertent perturbations.



## **18. Enhancing Adversarial Attacks on Single-Layer NVM Crossbar-Based Neural Networks with Power Consumption Information**

cs.LG

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2207.02764v1)

**Authors**: Cory Merkel

**Abstracts**: Adversarial attacks on state-of-the-art machine learning models pose a significant threat to the safety and security of mission-critical autonomous systems. This paper considers the additional vulnerability of machine learning models when attackers can measure the power consumption of their underlying hardware platform. In particular, we explore the utility of power consumption information for adversarial attacks on non-volatile memory crossbar-based single-layer neural networks. Our results from experiments with MNIST and CIFAR-10 datasets show that power consumption can reveal important information about the neural network's weight matrix, such as the 1-norm of its columns. That information can be used to infer the sensitivity of the network's loss with respect to different inputs. We also find that surrogate-based black box attacks that utilize crossbar power information can lead to improved attack efficiency.



## **19. LADDER: Latent Boundary-guided Adversarial Training**

cs.LG

To appear in Machine Learning

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2206.03717v2)

**Authors**: Xiaowei Zhou, Ivor W. Tsang, Jie Yin

**Abstracts**: Deep Neural Networks (DNNs) have recently achieved great success in many classification tasks. Unfortunately, they are vulnerable to adversarial attacks that generate adversarial examples with a small perturbation to fool DNN models, especially in model sharing scenarios. Adversarial training is proved to be the most effective strategy that injects adversarial examples into model training to improve the robustness of DNN models against adversarial attacks. However, adversarial training based on the existing adversarial examples fails to generalize well to standard, unperturbed test data. To achieve a better trade-off between standard accuracy and adversarial robustness, we propose a novel adversarial training framework called LAtent bounDary-guided aDvErsarial tRaining (LADDER) that adversarially trains DNN models on latent boundary-guided adversarial examples. As opposed to most of the existing methods that generate adversarial examples in the input space, LADDER generates a myriad of high-quality adversarial examples through adding perturbations to latent features. The perturbations are made along the normal of the decision boundary constructed by an SVM with an attention mechanism. We analyze the merits of our generated boundary-guided adversarial examples from a boundary field perspective and visualization view. Extensive experiments and detailed analysis on MNIST, SVHN, CelebA, and CIFAR-10 validate the effectiveness of LADDER in achieving a better trade-off between standard accuracy and adversarial robustness as compared with vanilla DNNs and competitive baselines.



## **20. Adversarial Robustness of Visual Dialog**

cs.CV

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2207.02639v1)

**Authors**: Lu Yu, Verena Rieser

**Abstracts**: Adversarial robustness evaluates the worst-case performance scenario of a machine learning model to ensure its safety and reliability. This study is the first to investigate the robustness of visually grounded dialog models towards textual attacks. These attacks represent a worst-case scenario where the input question contains a synonym which causes the previously correct model to return a wrong answer. Using this scenario, we first aim to understand how multimodal input components contribute to model robustness. Our results show that models which encode dialog history are more robust, and when launching an attack on history, model prediction becomes more uncertain. This is in contrast to prior work which finds that dialog history is negligible for model performance on this task. We also evaluate how to generate adversarial test examples which successfully fool the model but remain undetected by the user/software designer. We find that the textual, as well as the visual context are important to generate plausible worst-case scenarios.



## **21. Adversarial Mask: Real-World Universal Adversarial Attack on Face Recognition Models**

cs.CV

16

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2111.10759v2)

**Authors**: Alon Zolfi, Shai Avidan, Yuval Elovici, Asaf Shabtai

**Abstracts**: Deep learning-based facial recognition (FR) models have demonstrated state-of-the-art performance in the past few years, even when wearing protective medical face masks became commonplace during the COVID-19 pandemic. Given the outstanding performance of these models, the machine learning research community has shown increasing interest in challenging their robustness. Initially, researchers presented adversarial attacks in the digital domain, and later the attacks were transferred to the physical domain. However, in many cases, attacks in the physical domain are conspicuous, and thus may raise suspicion in real-world environments (e.g., airports). In this paper, we propose Adversarial Mask, a physical universal adversarial perturbation (UAP) against state-of-the-art FR models that is applied on face masks in the form of a carefully crafted pattern. In our experiments, we examined the transferability of our adversarial mask to a wide range of FR model architectures and datasets. In addition, we validated our adversarial mask's effectiveness in real-world experiments (CCTV use case) by printing the adversarial pattern on a fabric face mask. In these experiments, the FR system was only able to identify 3.34% of the participants wearing the mask (compared to a minimum of 83.34% with other evaluated masks). A demo of our experiments can be found at: https://youtu.be/_TXkDO5z11w.



## **22. FIDO2 With Two Displays-Or How to Protect Security-Critical Web Transactions Against Malware Attacks**

cs.CR

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2206.13358v3)

**Authors**: Timon Hackenjos, Benedikt Wagner, Julian Herr, Jochen Rill, Marek Wehmer, Niklas Goerke, Ingmar Baumgart

**Abstracts**: With the rise of attacks on online accounts in the past years, more and more services offer two-factor authentication for their users. Having factors out of two of the three categories something you know, something you have and something you are should ensure that an attacker cannot compromise two of them at once. Thus, an adversary should not be able to maliciously interact with one's account. However, this is only true if one considers a weak adversary. In particular, since most current solutions only authenticate a session and not individual transactions, they are noneffective if one's device is infected with malware. For online banking, the banking industry has long since identified the need for authenticating transactions. However, specifications of such authentication schemes are not public and implementation details vary wildly from bank to bank with most still being unable to protect against malware. In this work, we present a generic approach to tackle the problem of malicious account takeovers, even in the presence of malware. To this end, we define a new paradigm to improve two-factor authentication that involves the concepts of one-out-of-two security and transaction authentication. Web authentication schemes following this paradigm can protect security-critical transactions against manipulation, even if one of the factors is completely compromised. Analyzing existing authentication schemes, we find that they do not realize one-out-of-two security. We give a blueprint of how to design secure web authentication schemes in general. Based on this blueprint we propose FIDO2 With Two Displays (FIDO2D), a new web authentication scheme based on the FIDO2 standard and prove its security using Tamarin. We hope that our work inspires a new wave of more secure web authentication schemes, which protect security-critical transactions even against attacks with malware.



## **23. Learning to Accelerate Approximate Methods for Solving Integer Programming via Early Fixing**

cs.DM

16 pages, 11 figures, 6 tables

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2207.02087v1)

**Authors**: Longkang Li, Baoyuan Wu

**Abstracts**: Integer programming (IP) is an important and challenging problem. Approximate methods have shown promising performance on both effectiveness and efficiency for solving the IP problem. However, we observed that a large fraction of variables solved by some iterative approximate methods fluctuate around their final converged discrete states in very long iterations. Inspired by this observation, we aim to accelerate these approximate methods by early fixing these fluctuated variables to their converged states while not significantly harming the solution accuracy. To this end, we propose an early fixing framework along with the approximate method. We formulate the whole early fixing process as a Markov decision process, and train it using imitation learning. A policy network will evaluate the posterior probability of each free variable concerning its discrete candidate states in each block of iterations. Specifically, we adopt the powerful multi-headed attention mechanism in the policy network. Extensive experiments on our proposed early fixing framework are conducted to three different IP applications: constrained linear programming, MRF energy minimization and sparse adversarial attack. The former one is linear IP problem, while the latter two are quadratic IP problems. We extend the problem scale from regular size to significantly large size. The extensive experiments reveal the competitiveness of our early fixing framework: the runtime speeds up significantly, while the solution quality does not degrade much, even in some cases it is available to obtain better solutions. Our proposed early fixing framework can be regarded as an acceleration extension of ADMM methods for solving integer programming. The source codes are available at \url{https://github.com/SCLBD/Accelerated-Lpbox-ADMM}.



## **24. Benign Adversarial Attack: Tricking Models for Goodness**

cs.AI

ACM MM2022 Brave New Idea

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2107.11986v2)

**Authors**: Jitao Sang, Xian Zhao, Jiaming Zhang, Zhiyu Lin

**Abstracts**: In spite of the successful application in many fields, machine learning models today suffer from notorious problems like vulnerability to adversarial examples. Beyond falling into the cat-and-mouse game between adversarial attack and defense, this paper provides alternative perspective to consider adversarial example and explore whether we can exploit it in benign applications. We first attribute adversarial example to the human-model disparity on employing non-semantic features. While largely ignored in classical machine learning mechanisms, non-semantic feature enjoys three interesting characteristics as (1) exclusive to model, (2) critical to affect inference, and (3) utilizable as features. Inspired by this, we present brave new idea of benign adversarial attack to exploit adversarial examples for goodness in three directions: (1) adversarial Turing test, (2) rejecting malicious model application, and (3) adversarial data augmentation. Each direction is positioned with motivation elaboration, justification analysis and prototype applications to showcase its potential.



## **25. Formalizing and Estimating Distribution Inference Risks**

cs.LG

Update: Accepted at PETS 2022

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2109.06024v6)

**Authors**: Anshuman Suri, David Evans

**Abstracts**: Distribution inference, sometimes called property inference, infers statistical properties about a training set from access to a model trained on that data. Distribution inference attacks can pose serious risks when models are trained on private data, but are difficult to distinguish from the intrinsic purpose of statistical machine learning -- namely, to produce models that capture statistical properties about a distribution. Motivated by Yeom et al.'s membership inference framework, we propose a formal definition of distribution inference attacks that is general enough to describe a broad class of attacks distinguishing between possible training distributions. We show how our definition captures previous ratio-based property inference attacks as well as new kinds of attack including revealing the average node degree or clustering coefficient of a training graph. To understand distribution inference risks, we introduce a metric that quantifies observed leakage by relating it to the leakage that would occur if samples from the training distribution were provided directly to the adversary. We report on a series of experiments across a range of different distributions using both novel black-box attacks and improved versions of the state-of-the-art white-box attacks. Our results show that inexpensive attacks are often as effective as expensive meta-classifier attacks, and that there are surprising asymmetries in the effectiveness of attacks. Code is available at https://github.com/iamgroot42/FormEstDistRisks



## **26. Query-Efficient Adversarial Attack Based on Latin Hypercube Sampling**

cs.CV

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2207.02391v1)

**Authors**: Dan Wang, Jiayu Lin, Yuan-Gen Wang

**Abstracts**: In order to be applicable in real-world scenario, Boundary Attacks (BAs) were proposed and ensured one hundred percent attack success rate with only decision information. However, existing BA methods craft adversarial examples by leveraging a simple random sampling (SRS) to estimate the gradient, consuming a large number of model queries. To overcome the drawback of SRS, this paper proposes a Latin Hypercube Sampling based Boundary Attack (LHS-BA) to save query budget. Compared with SRS, LHS has better uniformity under the same limited number of random samples. Therefore, the average on these random samples is closer to the true gradient than that estimated by SRS. Various experiments are conducted on benchmark datasets including MNIST, CIFAR, and ImageNet-1K. Experimental results demonstrate the superiority of the proposed LHS-BA over the state-of-the-art BA methods in terms of query efficiency. The source codes are publicly available at https://github.com/GZHU-DVL/LHS-BA.



## **27. Task-agnostic Defense against Adversarial Patch Attacks**

cs.CV

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2207.01795v1)

**Authors**: Ke Xu, Yao Xiao, Zhaoheng Zheng, Kaijie Cai, Ram Nevatia

**Abstracts**: Adversarial patch attacks mislead neural networks by injecting adversarial pixels within a designated local region. Patch attacks can be highly effective in a variety of tasks and physically realizable via attachment (e.g. a sticker) to the real-world objects. Despite the diversity in attack patterns, adversarial patches tend to be highly textured and different in appearance from natural images. We exploit this property and present PatchZero, a task-agnostic defense against white-box adversarial patches. Specifically, our defense detects the adversarial pixels and "zeros out" the patch region by repainting with mean pixel values. We formulate the patch detection problem as a semantic segmentation task such that our model can generalize to patches of any size and shape. We further design a two-stage adversarial training scheme to defend against the stronger adaptive attacks. We thoroughly evaluate PatchZero on the image classification (ImageNet, RESISC45), object detection (PASCAL VOC), and video classification (UCF101) datasets. Our method achieves SOTA robust accuracy without any degradation in the benign performance.



## **28. Transferable Graph Backdoor Attack**

cs.CR

Accepted by the 25th International Symposium on Research in Attacks,  Intrusions, and Defenses

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2207.00425v3)

**Authors**: Shuiqiao Yang, Bao Gia Doan, Paul Montague, Olivier De Vel, Tamas Abraham, Seyit Camtepe, Damith C. Ranasinghe, Salil S. Kanhere

**Abstracts**: Graph Neural Networks (GNNs) have achieved tremendous success in many graph mining tasks benefitting from the message passing strategy that fuses the local structure and node features for better graph representation learning. Despite the success of GNNs, and similar to other types of deep neural networks, GNNs are found to be vulnerable to unnoticeable perturbations on both graph structure and node features. Many adversarial attacks have been proposed to disclose the fragility of GNNs under different perturbation strategies to create adversarial examples. However, vulnerability of GNNs to successful backdoor attacks was only shown recently. In this paper, we disclose the TRAP attack, a Transferable GRAPh backdoor attack. The core attack principle is to poison the training dataset with perturbation-based triggers that can lead to an effective and transferable backdoor attack. The perturbation trigger for a graph is generated by performing the perturbation actions on the graph structure via a gradient based score matrix from a surrogate model. Compared with prior works, TRAP attack is different in several ways: i) it exploits a surrogate Graph Convolutional Network (GCN) model to generate perturbation triggers for a blackbox based backdoor attack; ii) it generates sample-specific perturbation triggers which do not have a fixed pattern; and iii) the attack transfers, for the first time in the context of GNNs, to different GNN models when trained with the forged poisoned training dataset. Through extensive evaluations on four real-world datasets, we demonstrate the effectiveness of the TRAP attack to build transferable backdoors in four different popular GNNs using four real-world datasets.



## **29. Wild Networks: Exposure of 5G Network Infrastructures to Adversarial Examples**

cs.CR

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.01531v1)

**Authors**: Giovanni Apruzzese, Rodion Vladimirov, Aliya Tastemirova, Pavel Laskov

**Abstracts**: Fifth Generation (5G) networks must support billions of heterogeneous devices while guaranteeing optimal Quality of Service (QoS). Such requirements are impossible to meet with human effort alone, and Machine Learning (ML) represents a core asset in 5G. ML, however, is known to be vulnerable to adversarial examples; moreover, as our paper will show, the 5G context is exposed to a yet another type of adversarial ML attacks that cannot be formalized with existing threat models. Proactive assessment of such risks is also challenging due to the lack of ML-powered 5G equipment available for adversarial ML research.   To tackle these problems, we propose a novel adversarial ML threat model that is particularly suited to 5G scenarios, and is agnostic to the precise function solved by ML. In contrast to existing ML threat models, our attacks do not require any compromise of the target 5G system while still being viable due to the QoS guarantees and the open nature of 5G networks. Furthermore, we propose an original framework for realistic ML security assessments based on public data. We proactively evaluate our threat model on 6 applications of ML envisioned in 5G. Our attacks affect both the training and the inference stages, can degrade the performance of state-of-the-art ML systems, and have a lower entry barrier than previous attacks.



## **30. Adversarial Ensemble Training by Jointly Learning Label Dependencies and Member Models**

cs.LG

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2206.14477v2)

**Authors**: Lele Wang, Bin Liu

**Abstracts**: Training an ensemble of different sub-models has empirically proven to be an effective strategy to improve deep neural networks' adversarial robustness. Current ensemble training methods for image recognition usually encode the image labels by one-hot vectors, which neglect dependency relationships between the labels. Here we propose a novel adversarial ensemble training approach to jointly learn the label dependencies and the member models. Our approach adaptively exploits the learned label dependencies to promote the diversity of the member models. We test our approach on widely used datasets MNIST, FasionMNIST, and CIFAR-10. Results show that our approach is more robust against black-box attacks compared with the state-of-the-art methods. Our code is available at https://github.com/ZJLAB-AMMI/LSD.



## **31. Hessian-Free Second-Order Adversarial Examples for Adversarial Learning**

cs.LG

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.01396v1)

**Authors**: Yaguan Qian, Yuqi Wang, Bin Wang, Zhaoquan Gu, Yuhan Guo, Wassim Swaileh

**Abstracts**: Recent studies show deep neural networks (DNNs) are extremely vulnerable to the elaborately designed adversarial examples. Adversarial learning with those adversarial examples has been proved as one of the most effective methods to defend against such an attack. At present, most existing adversarial examples generation methods are based on first-order gradients, which can hardly further improve models' robustness, especially when facing second-order adversarial attacks. Compared with first-order gradients, second-order gradients provide a more accurate approximation of the loss landscape with respect to natural examples. Inspired by this, our work crafts second-order adversarial examples and uses them to train DNNs. Nevertheless, second-order optimization involves time-consuming calculation for Hessian-inverse. We propose an approximation method through transforming the problem into an optimization in the Krylov subspace, which remarkably reduce the computational complexity to speed up the training procedure. Extensive experiments conducted on the MINIST and CIFAR-10 datasets show that our adversarial learning with second-order adversarial examples outperforms other fisrt-order methods, which can improve the model robustness against a wide range of attacks.



## **32. Training strategy for a lightweight countermeasure model for automatic speaker verification**

cs.SD

Need to finish experiment

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2203.17031v4)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstracts**: The countermeasure (CM) model is developed to protect Automatic Speaker Verification (ASV) systems from spoof attacks and prevent resulting personal information leakage. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud-based systems. This work proposes training strategies for a lightweight CM model for ASV, using generalized end-to-end (GE2E) pre-training and adversarial fine-tuning to improve performance, and applying knowledge distillation (KD) to reduce the size of the CM model. In the evaluation phase of the ASVspoof 2021 Logical Access task, the lightweight ResNetSE model reaches min t-DCF 0.2695 and EER 3.54%. Compared to the teacher model, the lightweight student model only uses 22.5% of parameters and 21.1% of multiply and accumulate operands of the teacher model.



## **33. BadHash: Invisible Backdoor Attacks against Deep Hashing with Clean Label**

cs.CV

conference

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.00278v2)

**Authors**: Shengshan Hu, Ziqi Zhou, Yechao Zhang, Leo Yu Zhang, Yifeng Zheng, Yuanyuan HE, Hai Jin

**Abstracts**: Due to its powerful feature learning capability and high efficiency, deep hashing has achieved great success in large-scale image retrieval. Meanwhile, extensive works have demonstrated that deep neural networks (DNNs) are susceptible to adversarial examples, and exploring adversarial attack against deep hashing has attracted many research efforts. Nevertheless, backdoor attack, another famous threat to DNNs, has not been studied for deep hashing yet. Although various backdoor attacks have been proposed in the field of image classification, existing approaches failed to realize a truly imperceptive backdoor attack that enjoys invisible triggers and clean label setting simultaneously, and they also cannot meet the intrinsic demand of image retrieval backdoor.   In this paper, we propose BadHash, the first generative-based imperceptible backdoor attack against deep hashing, which can effectively generate invisible and input-specific poisoned images with clean label. Specifically, we first propose a new conditional generative adversarial network (cGAN) pipeline to effectively generate poisoned samples. For any given benign image, it seeks to generate a natural-looking poisoned counterpart with a unique invisible trigger. In order to improve the attack effectiveness, we introduce a label-based contrastive learning network LabCLN to exploit the semantic characteristics of different labels, which are subsequently used for confusing and misleading the target model to learn the embedded trigger. We finally explore the mechanism of backdoor attacks on image retrieval in the hash space. Extensive experiments on multiple benchmark datasets verify that BadHash can generate imperceptible poisoned samples with strong attack ability and transferability over state-of-the-art deep hashing schemes.



## **34. Removing Batch Normalization Boosts Adversarial Training**

cs.LG

ICML 2022

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.01156v1)

**Authors**: Haotao Wang, Aston Zhang, Shuai Zheng, Xingjian Shi, Mu Li, Zhangyang Wang

**Abstracts**: Adversarial training (AT) defends deep neural networks against adversarial attacks. One challenge that limits its practical application is the performance degradation on clean samples. A major bottleneck identified by previous works is the widely used batch normalization (BN), which struggles to model the different statistics of clean and adversarial training samples in AT. Although the dominant approach is to extend BN to capture this mixture of distribution, we propose to completely eliminate this bottleneck by removing all BN layers in AT. Our normalizer-free robust training (NoFrost) method extends recent advances in normalizer-free networks to AT for its unexplored advantage on handling the mixture distribution challenge. We show that NoFrost achieves adversarial robustness with only a minor sacrifice on clean sample accuracy. On ImageNet with ResNet50, NoFrost achieves $74.06\%$ clean accuracy, which drops merely $2.00\%$ from standard training. In contrast, BN-based AT obtains $59.28\%$ clean accuracy, suffering a significant $16.78\%$ drop from standard training. In addition, NoFrost achieves a $23.56\%$ adversarial robustness against PGD attack, which improves the $13.57\%$ robustness in BN-based AT. We observe better model smoothness and larger decision margins from NoFrost, which make the models less sensitive to input perturbations and thus more robust. Moreover, when incorporating more data augmentations into NoFrost, it achieves comprehensive robustness against multiple distribution shifts. Code and pre-trained models are public at https://github.com/amazon-research/normalizer-free-robust-training.



## **35. RAF: Recursive Adversarial Attacks on Face Recognition Using Extremely Limited Queries**

cs.CV

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.01149v1)

**Authors**: Keshav Kasichainula, Hadi Mansourifar, Weidong Shi

**Abstracts**: Recent successful adversarial attacks on face recognition show that, despite the remarkable progress of face recognition models, they are still far behind the human intelligence for perception and recognition. It reveals the vulnerability of deep convolutional neural networks (CNNs) as state-of-the-art building block for face recognition models against adversarial examples, which can cause certain consequences for secure systems. Gradient-based adversarial attacks are widely studied before and proved to be successful against face recognition models. However, finding the optimized perturbation per each face needs to submitting the significant number of queries to the target model. In this paper, we propose recursive adversarial attack on face recognition using automatic face warping which needs extremely limited number of queries to fool the target model. Instead of a random face warping procedure, the warping functions are applied on specific detected regions of face like eyebrows, nose, lips, etc. We evaluate the robustness of proposed method in the decision-based black-box attack setting, where the attackers have no access to the model parameters and gradients, but hard-label predictions and confidence scores are provided by the target model.



## **36. Generating gender-ambiguous voices for privacy-preserving speech recognition**

cs.SD

5 pages, 4 figures, submitted to INTERSPEECH

**SubmitDate**: 2022-07-03    [paper-pdf](http://arxiv.org/pdf/2207.01052v1)

**Authors**: Dimitrios Stoidis, Andrea Cavallaro

**Abstracts**: Our voice encodes a uniquely identifiable pattern which can be used to infer private attributes, such as gender or identity, that an individual might wish not to reveal when using a speech recognition service. To prevent attribute inference attacks alongside speech recognition tasks, we present a generative adversarial network, GenGAN, that synthesises voices that conceal the gender or identity of a speaker. The proposed network includes a generator with a U-Net architecture that learns to fool a discriminator. We condition the generator only on gender information and use an adversarial loss between signal distortion and privacy preservation. We show that GenGAN improves the trade-off between privacy and utility compared to privacy-preserving representation learning methods that consider gender information as a sensitive attribute to protect.



## **37. Adversarial Attack and Defense of YOLO Detectors in Autonomous Driving Scenarios**

cs.CV

Accepted by 2022 IEEE Intelligent Vehicles Symposium (IV 2022)

**SubmitDate**: 2022-07-03    [paper-pdf](http://arxiv.org/pdf/2202.04781v2)

**Authors**: Jung Im Choi, Qing Tian

**Abstracts**: Visual detection is a key task in autonomous driving, and it serves as a crucial foundation for self-driving planning and control. Deep neural networks have achieved promising results in various visual tasks, but they are known to be vulnerable to adversarial attacks. A comprehensive understanding of deep visual detectors' vulnerability is required before people can improve their robustness. However, only a few adversarial attack/defense works have focused on object detection, and most of them employed only classification and/or localization losses, ignoring the objectness aspect. In this paper, we identify a serious objectness-related adversarial vulnerability in YOLO detectors and present an effective attack strategy targeting the objectness aspect of visual detection in autonomous vehicles. Furthermore, to address such vulnerability, we propose a new objectness-aware adversarial training approach for visual detection. Experiments show that the proposed attack targeting the objectness aspect is 45.17% and 43.50% more effective than those generated from classification and/or localization losses on the KITTI and COCO traffic datasets, respectively. Also, the proposed adversarial defense approach can improve the detectors' robustness against objectness-oriented attacks by up to 21% and 12% mAP on KITTI and COCO traffic, respectively.



## **38. AGIC: Approximate Gradient Inversion Attack on Federated Learning**

cs.LG

This paper is accepted at the 41st International Symposium on  Reliable Distributed Systems (SRDS 2022)

**SubmitDate**: 2022-07-03    [paper-pdf](http://arxiv.org/pdf/2204.13784v2)

**Authors**: Jin Xu, Chi Hong, Jiyue Huang, Lydia Y. Chen, Jérémie Decouchant

**Abstracts**: Federated learning is a private-by-design distributed learning paradigm where clients train local models on their own data before a central server aggregates their local updates to compute a global model. Depending on the aggregation method used, the local updates are either the gradients or the weights of local learning models. Recent reconstruction attacks apply a gradient inversion optimization on the gradient update of a single minibatch to reconstruct the private data used by clients during training. As the state-of-the-art reconstruction attacks solely focus on single update, realistic adversarial scenarios are overlooked, such as observation across multiple updates and updates trained from multiple mini-batches. A few studies consider a more challenging adversarial scenario where only model updates based on multiple mini-batches are observable, and resort to computationally expensive simulation to untangle the underlying samples for each local step. In this paper, we propose AGIC, a novel Approximate Gradient Inversion Attack that efficiently and effectively reconstructs images from both model or gradient updates, and across multiple epochs. In a nutshell, AGIC (i) approximates gradient updates of used training samples from model updates to avoid costly simulation procedures, (ii) leverages gradient/model updates collected from multiple epochs, and (iii) assigns increasing weights to layers with respect to the neural network structure for reconstruction quality. We extensively evaluate AGIC on three datasets, CIFAR-10, CIFAR-100 and ImageNet. Our results show that AGIC increases the peak signal-to-noise ratio (PSNR) by up to 50% compared to two representative state-of-the-art gradient inversion attacks. Furthermore, AGIC is faster than the state-of-the-art simulation based attack, e.g., it is 5x faster when attacking FedAvg with 8 local steps in between model updates.



## **39. Tricking the Hashing Trick: A Tight Lower Bound on the Robustness of CountSketch to Adaptive Inputs**

cs.DS

**SubmitDate**: 2022-07-03    [paper-pdf](http://arxiv.org/pdf/2207.00956v1)

**Authors**: Edith Cohen, Jelani Nelson, Tamás Sarlós, Uri Stemmer

**Abstracts**: CountSketch and Feature Hashing (the "hashing trick") are popular randomized dimensionality reduction methods that support recovery of $\ell_2$-heavy hitters (keys $i$ where $v_i^2 > \epsilon \|\boldsymbol{v}\|_2^2$) and approximate inner products. When the inputs are {\em not adaptive} (do not depend on prior outputs), classic estimators applied to a sketch of size $O(\ell/\epsilon)$ are accurate for a number of queries that is exponential in $\ell$. When inputs are adaptive, however, an adversarial input can be constructed after $O(\ell)$ queries with the classic estimator and the best known robust estimator only supports $\tilde{O}(\ell^2)$ queries. In this work we show that this quadratic dependence is in a sense inherent: We design an attack that after $O(\ell^2)$ queries produces an adversarial input vector whose sketch is highly biased. Our attack uses "natural" non-adaptive inputs (only the final adversarial input is chosen adaptively) and universally applies with any correct estimator, including one that is unknown to the attacker. In that, we expose inherent vulnerability of this fundamental method.



## **40. When Are Linear Stochastic Bandits Attackable?**

cs.LG

27 pages, 3 figures, ICML 2022

**SubmitDate**: 2022-07-03    [paper-pdf](http://arxiv.org/pdf/2110.09008v2)

**Authors**: Huazheng Wang, Haifeng Xu, Hongning Wang

**Abstracts**: We study adversarial attacks on linear stochastic bandits: by manipulating the rewards, an adversary aims to control the behaviour of the bandit algorithm. Perhaps surprisingly, we first show that some attack goals can never be achieved. This is in sharp contrast to context-free stochastic bandits, and is intrinsically due to the correlation among arms in linear stochastic bandits. Motivated by this finding, this paper studies the attackability of a $k$-armed linear bandit environment. We first provide a complete necessity and sufficiency characterization of attackability based on the geometry of the arms' context vectors. We then propose a two-stage attack method against LinUCB and Robust Phase Elimination. The method first asserts whether the given environment is attackable; and if yes, it poisons the rewards to force the algorithm to pull a target arm linear times using only a sublinear cost. Numerical experiments further validate the effectiveness and cost-efficiency of the proposed attack method.



## **41. Certifiably Robust Policy Learning against Adversarial Communication in Multi-agent Systems**

cs.LG

**SubmitDate**: 2022-07-02    [paper-pdf](http://arxiv.org/pdf/2206.10158v2)

**Authors**: Yanchao Sun, Ruijie Zheng, Parisa Hassanzadeh, Yongyuan Liang, Soheil Feizi, Sumitra Ganesh, Furong Huang

**Abstracts**: Communication is important in many multi-agent reinforcement learning (MARL) problems for agents to share information and make good decisions. However, when deploying trained communicative agents in a real-world application where noise and potential attackers exist, the safety of communication-based policies becomes a severe issue that is underexplored. Specifically, if communication messages are manipulated by malicious attackers, agents relying on untrustworthy communication may take unsafe actions that lead to catastrophic consequences. Therefore, it is crucial to ensure that agents will not be misled by corrupted communication, while still benefiting from benign communication. In this work, we consider an environment with $N$ agents, where the attacker may arbitrarily change the communication from any $C<\frac{N-1}{2}$ agents to a victim agent. For this strong threat model, we propose a certifiable defense by constructing a message-ensemble policy that aggregates multiple randomly ablated message sets. Theoretical analysis shows that this message-ensemble policy can utilize benign communication while being certifiably robust to adversarial communication, regardless of the attacking algorithm. Experiments in multiple environments verify that our defense significantly improves the robustness of trained policies against various types of attacks.



## **42. Backdoor Attack is A Devil in Federated GAN-based Medical Image Synthesis**

cs.CV

13 pages, 4 figures

**SubmitDate**: 2022-07-02    [paper-pdf](http://arxiv.org/pdf/2207.00762v1)

**Authors**: Ruinan Jin, Xiaoxiao Li

**Abstracts**: Deep Learning-based image synthesis techniques have been applied in healthcare research for generating medical images to support open research. Training generative adversarial neural networks (GAN) usually requires large amounts of training data. Federated learning (FL) provides a way of training a central model using distributed data from different medical institutions while keeping raw data locally. However, FL is vulnerable to backdoor attack, an adversarial by poisoning training data, given the central server cannot access the original data directly. Most backdoor attack strategies focus on classification models and centralized domains. In this study, we propose a way of attacking federated GAN (FedGAN) by treating the discriminator with a commonly used data poisoning strategy in backdoor attack classification models. We demonstrate that adding a small trigger with size less than 0.5 percent of the original image size can corrupt the FL-GAN model. Based on the proposed attack, we provide two effective defense strategies: global malicious detection and local training regularization. We show that combining the two defense strategies yields a robust medical image generation.



## **43. Efficient Adversarial Training With Data Pruning**

cs.LG

**SubmitDate**: 2022-07-01    [paper-pdf](http://arxiv.org/pdf/2207.00694v1)

**Authors**: Maximilian Kaufmann, Yiren Zhao, Ilia Shumailov, Robert Mullins, Nicolas Papernot

**Abstracts**: Neural networks are susceptible to adversarial examples-small input perturbations that cause models to fail. Adversarial training is one of the solutions that stops adversarial examples; models are exposed to attacks during training and learn to be resilient to them. Yet, such a procedure is currently expensive-it takes a long time to produce and train models with adversarial samples, and, what is worse, it occasionally fails. In this paper we demonstrate data pruning-a method for increasing adversarial training efficiency through data sub-sampling.We empirically show that data pruning leads to improvements in convergence and reliability of adversarial training, albeit with different levels of utility degradation. For example, we observe that using random sub-sampling of CIFAR10 to drop 40% of data, we lose 8% adversarial accuracy against the strongest attackers, while by using only 20% of data we lose 14% adversarial accuracy and reduce runtime by a factor of 3. Interestingly, we discover that in some settings data pruning brings benefits from both worlds-it both improves adversarial accuracy and training time.



## **44. Watermarking Graph Neural Networks based on Backdoor Attacks**

cs.LG

13 pages, 9 figures

**SubmitDate**: 2022-07-01    [paper-pdf](http://arxiv.org/pdf/2110.11024v3)

**Authors**: Jing Xu, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. Building a powerful GNN model is not a trivial task, as it requires a large amount of training data, powerful computing resources, and human expertise in fine-tuning the model. What is more, with the development of adversarial attacks, e.g., model stealing attacks, GNNs raise challenges to model authentication. To avoid copyright infringement on GNNs, it is necessary to verify the ownership of the GNN models.   In this paper, we present a watermarking framework for GNNs for both graph and node classification tasks. We 1) design two strategies to generate watermarked data for the graph classification task and one for the node classification task, 2) embed the watermark into the host model through training to obtain the watermarked GNN model, and 3) verify the ownership of the suspicious model in a black-box setting. The experiments show that our framework can verify the ownership of GNN models with a very high probability (around $95\%$) for both tasks. Finally, we experimentally show that our watermarking approach is robust against two model modifications and an input reformation defense against backdoor attacks.



## **45. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

cs.CR

15 pages, 15 figures

**SubmitDate**: 2022-07-01    [paper-pdf](http://arxiv.org/pdf/2202.03195v3)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for a different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against two state-of-the-art defenses. We find that both attacks are robust against the investigated defenses, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.



## **46. Effect of Homomorphic Encryption on the Performance of Training Federated Learning Generative Adversarial Networks**

cs.CR

**SubmitDate**: 2022-07-01    [paper-pdf](http://arxiv.org/pdf/2207.00263v1)

**Authors**: Ignjat Pejic, Rui Wang, Kaitai Liang

**Abstracts**: A Generative Adversarial Network (GAN) is a deep-learning generative model in the field of Machine Learning (ML) that involves training two Neural Networks (NN) using a sizable data set. In certain fields, such as medicine, the training data may be hospital patient records that are stored across different hospitals. The classic centralized approach would involve sending the data to a centralized server where the model would be trained. However, that would involve breaching the privacy and confidentiality of the patients and their data, which would be unacceptable. Therefore, Federated Learning (FL), an ML technique that trains ML models in a distributed setting without data ever leaving the host device, would be a better alternative to the centralized option. In this ML technique, only parameters and certain metadata would be communicated. In spite of that, there still exist attacks that can infer user data using the parameters and metadata. A fully privacy-preserving solution involves homomorphically encrypting (HE) the data communicated. This paper will focus on the performance loss of training an FL-GAN with three different types of Homomorphic Encryption: Partial Homomorphic Encryption (PHE), Somewhat Homomorphic Encryption (SHE), and Fully Homomorphic Encryption (FHE). We will also test the performance loss of Multi-Party Computations (MPC), as it has homomorphic properties. The performances will be compared to the performance of training an FL-GAN without encryption as well. Our experiments show that the more complex the encryption method is, the longer it takes, with the extra time taken for HE is quite significant in comparison to the base case of FL.



## **47. Threat Assessment in Machine Learning based Systems**

cs.CR

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2207.00091v1)

**Authors**: Lionel Nganyewou Tidjon, Foutse Khomh

**Abstracts**: Machine learning is a field of artificial intelligence (AI) that is becoming essential for several critical systems, making it a good target for threat actors. Threat actors exploit different Tactics, Techniques, and Procedures (TTPs) against the confidentiality, integrity, and availability of Machine Learning (ML) systems. During the ML cycle, they exploit adversarial TTPs to poison data and fool ML-based systems. In recent years, multiple security practices have been proposed for traditional systems but they are not enough to cope with the nature of ML-based systems. In this paper, we conduct an empirical study of threats reported against ML-based systems with the aim to understand and characterize the nature of ML threats and identify common mitigation strategies. The study is based on 89 real-world ML attack scenarios from the MITRE's ATLAS database, the AI Incident Database, and the literature; 854 ML repositories from the GitHub search and the Python Packaging Advisory database, selected based on their reputation. Attacks from the AI Incident Database and the literature are used to identify vulnerabilities and new types of threats that were not documented in ATLAS. Results show that convolutional neural networks were one of the most targeted models among the attack scenarios. ML repositories with the largest vulnerability prominence include TensorFlow, OpenCV, and Notebook. In this paper, we also report the most frequent vulnerabilities in the studied ML repositories, the most targeted ML phases and models, the most used TTPs in ML phases and attack scenarios. This information is particularly important for red/blue teams to better conduct attacks/defenses, for practitioners to prevent threats during ML development, and for researchers to develop efficient defense mechanisms.



## **48. MEAD: A Multi-Armed Approach for Evaluation of Adversarial Examples Detectors**

cs.CV

This paper has been accepted to appear in the Proceedings of the 2022  European Conference on Machine Learning and Data Mining (ECML-PKDD), 19th to  the 23rd of September, Grenoble, France

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2206.15415v1)

**Authors**: Federica Granese, Marine Picot, Marco Romanelli, Francisco Messina, Pablo Piantanida

**Abstracts**: Detection of adversarial examples has been a hot topic in the last years due to its importance for safely deploying machine learning algorithms in critical applications. However, the detection methods are generally validated by assuming a single implicitly known attack strategy, which does not necessarily account for real-life threats. Indeed, this can lead to an overoptimistic assessment of the detectors' performance and may induce some bias in the comparison between competing detection schemes. We propose a novel multi-armed framework, called MEAD, for evaluating detectors based on several attack strategies to overcome this limitation. Among them, we make use of three new objectives to generate attacks. The proposed performance metric is based on the worst-case scenario: detection is successful if and only if all different attacks are correctly recognized. Empirically, we show the effectiveness of our approach. Moreover, the poor performance obtained for state-of-the-art detectors opens a new exciting line of research.



## **49. The Topological BERT: Transforming Attention into Topology for Natural Language Processing**

cs.CL

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2206.15195v1)

**Authors**: Ilan Perez, Raphael Reinauer

**Abstracts**: In recent years, the introduction of the Transformer models sparked a revolution in natural language processing (NLP). BERT was one of the first text encoders using only the attention mechanism without any recurrent parts to achieve state-of-the-art results on many NLP tasks.   This paper introduces a text classifier using topological data analysis. We use BERT's attention maps transformed into attention graphs as the only input to that classifier. The model can solve tasks such as distinguishing spam from ham messages, recognizing whether a sentence is grammatically correct, or evaluating a movie review as negative or positive. It performs comparably to the BERT baseline and outperforms it on some tasks.   Additionally, we propose a new method to reduce the number of BERT's attention heads considered by the topological classifier, which allows us to prune the number of heads from 144 down to as few as ten with no reduction in performance. Our work also shows that the topological model displays higher robustness against adversarial attacks than the original BERT model, which is maintained during the pruning process. To the best of our knowledge, this work is the first to confront topological-based models with adversarial attacks in the context of NLP.



## **50. Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN**

cs.LG

Accepted in KDD2022

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2207.00012v1)

**Authors**: Kuan Li, Yang Liu, Xiang Ao, Jianfeng Chi, Jinghua Feng, Hao Yang, Qing He

**Abstracts**: Benefiting from the message passing mechanism, Graph Neural Networks (GNNs) have been successful on flourish tasks over graph data. However, recent studies have shown that attackers can catastrophically degrade the performance of GNNs by maliciously modifying the graph structure. A straightforward solution to remedy this issue is to model the edge weights by learning a metric function between pairwise representations of two end nodes, which attempts to assign low weights to adversarial edges. The existing methods use either raw features or representations learned by supervised GNNs to model the edge weights. However, both strategies are faced with some immediate problems: raw features cannot represent various properties of nodes (e.g., structure information), and representations learned by supervised GNN may suffer from the poor performance of the classifier on the poisoned graph. We need representations that carry both feature information and as mush correct structure information as possible and are insensitive to structural perturbations. To this end, we propose an unsupervised pipeline, named STABLE, to optimize the graph structure. Finally, we input the well-refined graph into a downstream classifier. For this part, we design an advanced GCN that significantly enhances the robustness of vanilla GCN without increasing the time complexity. Extensive experiments on four real-world graph benchmarks demonstrate that STABLE outperforms the state-of-the-art methods and successfully defends against various attacks.



