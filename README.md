# Latest Adversarial Attack Papers
**update at 2022-08-16 16:52:12**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. MENLI: Robust Evaluation Metrics from Natural Language Inference**

cs.CL

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.07316v1)

**Authors**: Yanran Chen, Steffen Eger

**Abstracts**: Recently proposed BERT-based evaluation metrics perform well on standard evaluation benchmarks but are vulnerable to adversarial attacks, e.g., relating to factuality errors. We argue that this stems (in part) from the fact that they are models of semantic similarity. In contrast, we develop evaluation metrics based on Natural Language Inference (NLI), which we deem a more appropriate modeling. We design a preference-based adversarial attack framework and show that our NLI based metrics are much more robust to the attacks than the recent BERT-based metrics. On standard benchmarks, our NLI based metrics outperform existing summarization metrics, but perform below SOTA MT metrics. However, when we combine existing metrics with our NLI metrics, we obtain both higher adversarial robustness (+20% to +30%) and higher quality metrics as measured on standard benchmarks (+5% to +25%).



## **2. Bounding Membership Inference**

cs.LG

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2202.12232v3)

**Authors**: Anvith Thudi, Ilia Shumailov, Franziska Boenisch, Nicolas Papernot

**Abstracts**: Differential Privacy (DP) is the de facto standard for reasoning about the privacy guarantees of a training algorithm. Despite the empirical observation that DP reduces the vulnerability of models to existing membership inference (MI) attacks, a theoretical underpinning as to why this is the case is largely missing in the literature. In practice, this means that models need to be trained with DP guarantees that greatly decrease their accuracy. In this paper, we provide a tighter bound on the positive accuracy (i.e., attack precision) of any MI adversary when a training algorithm provides $\epsilon$-DP or $(\epsilon, \delta)$-DP. Our bound informs the design of a novel privacy amplification scheme, where an effective training set is sub-sampled from a larger set prior to the beginning of training, to greatly reduce the bound on MI accuracy. As a result, our scheme enables DP users to employ looser DP guarantees when training their model to limit the success of any MI adversary; this ensures that the model's accuracy is less impacted by the privacy guarantee. Finally, we discuss implications of our MI bound on the field of machine unlearning.



## **3. Man-in-the-Middle Attack against Object Detection Systems**

cs.RO

7 pages, 8 figures

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.07174v1)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstracts**: Is deep learning secure for robots? As embedded systems have access to more powerful CPUs and GPUs, deep-learning-enabled object detection systems become pervasive in robotic applications. Meanwhile, prior research unveils that deep learning models are vulnerable to adversarial attacks. Does this put real-world robots at threat? Our research borrows the idea of the Main-in-the-Middle attack from Cryptography to attack an object detection system. Our experimental results prove that we can generate a strong Universal Adversarial Perturbation (UAP) within one minute and then use the perturbation to attack a detection system via the Man-in-the-Middle attack. Our findings raise a serious concern over the applications of deep learning models in safety-critical systems such as autonomous driving.



## **4. GUARD: Graph Universal Adversarial Defense**

cs.LG

Preprint. Code is publicly available at  https://github.com/EdisonLeeeee/GUARD

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2204.09803v3)

**Authors**: Jintang Li, Jie Liao, Ruofan Wu, Liang Chen, Jiawang Dan, Changhua Meng, Zibin Zheng, Weiqiang Wang

**Abstracts**: Graph convolutional networks (GCNs) have shown to be vulnerable to small adversarial perturbations, which becomes a severe threat and largely limits their applications in security-critical scenarios. To mitigate such a threat, considerable research efforts have been devoted to increasing the robustness of GCNs against adversarial attacks. However, current approaches for defense are typically designed for the whole graph and consider the global performance, posing challenges in protecting important local nodes from stronger adversarial targeted attacks. In this work, we present a simple yet effective method, named Graph Universal Adversarial Defense (GUARD). Unlike previous works, GUARD protects each individual node from attacks with a universal defensive patch, which is generated once and can be applied to any node (node-agnostic) in a graph. Extensive experiments on four benchmark datasets demonstrate that our method significantly improves robustness for several established GCNs against multiple adversarial attacks and outperforms state-of-the-art defense methods by large margins. Our code is publicly available at https://github.com/EdisonLeeeee/GUARD.



## **5. A Multi-objective Memetic Algorithm for Auto Adversarial Attack Optimization Design**

cs.CV

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.06984v1)

**Authors**: Jialiang Sun, Wen Yao, Tingsong Jiang, Xiaoqian Chen

**Abstracts**: The phenomenon of adversarial examples has been revealed in variant scenarios. Recent studies show that well-designed adversarial defense strategies can improve the robustness of deep learning models against adversarial examples. However, with the rapid development of defense technologies, it also tends to be more difficult to evaluate the robustness of the defensed model due to the weak performance of existing manually designed adversarial attacks. To address the challenge, given the defensed model, the efficient adversarial attack with less computational burden and lower robust accuracy is needed to be further exploited. Therefore, we propose a multi-objective memetic algorithm for auto adversarial attack optimization design, which realizes the automatical search for the near-optimal adversarial attack towards defensed models. Firstly, the more general mathematical model of auto adversarial attack optimization design is constructed, where the search space includes not only the attacker operations, magnitude, iteration number, and loss functions but also the connection ways of multiple adversarial attacks. In addition, we develop a multi-objective memetic algorithm combining NSGA-II and local search to solve the optimization problem. Finally, to decrease the evaluation cost during the search, we propose a representative data selection strategy based on the sorting of cross entropy loss values of each images output by models. Experiments on CIFAR10, CIFAR100, and ImageNet datasets show the effectiveness of our proposed method.



## **6. InvisibiliTee: Angle-agnostic Cloaking from Person-Tracking Systems with a Tee**

cs.CV

12 pages, 10 figures and the ICANN 2022 accpeted paper

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.06962v1)

**Authors**: Yaxian Li, Bingqing Zhang, Guoping Zhao, Mingyu Zhang, Jiajun Liu, Ziwei Wang, Jirong Wen

**Abstracts**: After a survey for person-tracking system-induced privacy concerns, we propose a black-box adversarial attack method on state-of-the-art human detection models called InvisibiliTee. The method learns printable adversarial patterns for T-shirts that cloak wearers in the physical world in front of person-tracking systems. We design an angle-agnostic learning scheme which utilizes segmentation of the fashion dataset and a geometric warping process so the adversarial patterns generated are effective in fooling person detectors from all camera angles and for unseen black-box detection models. Empirical results in both digital and physical environments show that with the InvisibiliTee on, person-tracking systems' ability to detect the wearer drops significantly.



## **7. ARIEL: Adversarial Graph Contrastive Learning**

cs.LG

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.06956v1)

**Authors**: Shengyu Feng, Baoyu Jing, Yada Zhu, Hanghang Tong

**Abstracts**: Contrastive learning is an effective unsupervised method in graph representation learning, and the key component of contrastive learning lies in the construction of positive and negative samples. Previous methods usually utilize the proximity of nodes in the graph as the principle. Recently, the data augmentation based contrastive learning method has advanced to show great power in the visual domain, and some works extended this method from images to graphs. However, unlike the data augmentation on images, the data augmentation on graphs is far less intuitive and much harder to provide high-quality contrastive samples, which leaves much space for improvement. In this work, by introducing an adversarial graph view for data augmentation, we propose a simple but effective method, Adversarial Graph Contrastive Learning (ARIEL), to extract informative contrastive samples within reasonable constraints. We develop a new technique called information regularization for stable training and use subgraph sampling for scalability. We generalize our method from node-level contrastive learning to the graph-level by treating each graph instance as a supernode. ARIEL consistently outperforms the current graph contrastive learning methods for both node-level and graph-level classification tasks on real-world datasets. We further demonstrate that ARIEL is more robust in face of adversarial attacks.



## **8. GNPassGAN: Improved Generative Adversarial Networks For Trawling Offline Password Guessing**

cs.CR

9 pages, 8 tables, 3 figures

**SubmitDate**: 2022-08-14    [paper-pdf](http://arxiv.org/pdf/2208.06943v1)

**Authors**: Fangyi Yu, Miguel Vargas Martin

**Abstracts**: The security of passwords depends on a thorough understanding of the strategies used by attackers. Unfortunately, real-world adversaries use pragmatic guessing tactics like dictionary attacks, which are difficult to simulate in password security research. Dictionary attacks must be carefully configured and modified to represent an actual threat. This approach, however, needs domain-specific knowledge and expertise that are difficult to duplicate. This paper reviews various deep learning-based password guessing approaches that do not require domain knowledge or assumptions about users' password structures and combinations. It also introduces GNPassGAN, a password guessing tool built on generative adversarial networks for trawling offline attacks. In comparison to the state-of-the-art PassGAN model, GNPassGAN is capable of guessing 88.03\% more passwords and generating 31.69\% fewer duplicates.



## **9. Gradient Mask: Lateral Inhibition Mechanism Improves Performance in Artificial Neural Networks**

cs.CV

**SubmitDate**: 2022-08-14    [paper-pdf](http://arxiv.org/pdf/2208.06918v1)

**Authors**: Lei Jiang, Yongqing Liu, Shihai Xiao, Yansong Chua

**Abstracts**: Lateral inhibitory connections have been observed in the cortex of the biological brain, and has been extensively studied in terms of its role in cognitive functions. However, in the vanilla version of backpropagation in deep learning, all gradients (which can be understood to comprise of both signal and noise gradients) flow through the network during weight updates. This may lead to overfitting. In this work, inspired by biological lateral inhibition, we propose Gradient Mask, which effectively filters out noise gradients in the process of backpropagation. This allows the learned feature information to be more intensively stored in the network while filtering out noisy or unimportant features. Furthermore, we demonstrate analytically how lateral inhibition in artificial neural networks improves the quality of propagated gradients. A new criterion for gradient quality is proposed which can be used as a measure during training of various convolutional neural networks (CNNs). Finally, we conduct several different experiments to study how Gradient Mask improves the performance of the network both quantitatively and qualitatively. Quantitatively, accuracy in the original CNN architecture, accuracy after pruning, and accuracy after adversarial attacks have shown improvements. Qualitatively, the CNN trained using Gradient Mask has developed saliency maps that focus primarily on the object of interest, which is useful for data augmentation and network interpretability.



## **10. IPvSeeYou: Exploiting Leaked Identifiers in IPv6 for Street-Level Geolocation**

cs.NI

Accepted to S&P '23

**SubmitDate**: 2022-08-14    [paper-pdf](http://arxiv.org/pdf/2208.06767v1)

**Authors**: Erik Rye, Robert Beverly

**Abstracts**: We present IPvSeeYou, a privacy attack that permits a remote and unprivileged adversary to physically geolocate many residential IPv6 hosts and networks with street-level precision. The crux of our method involves: 1) remotely discovering wide area (WAN) hardware MAC addresses from home routers; 2) correlating these MAC addresses with their WiFi BSSID counterparts of known location; and 3) extending coverage by associating devices connected to a common penultimate provider router.   We first obtain a large corpus of MACs embedded in IPv6 addresses via high-speed network probing. These MAC addresses are effectively leaked up the protocol stack and largely represent WAN interfaces of residential routers, many of which are all-in-one devices that also provide WiFi. We develop a technique to statistically infer the mapping between a router's WAN and WiFi MAC addresses across manufacturers and devices, and mount a large-scale data fusion attack that correlates WAN MACs with WiFi BSSIDs available in wardriving (geolocation) databases. Using these correlations, we geolocate the IPv6 prefixes of $>$12M routers in the wild across 146 countries and territories. Selected validation confirms a median geolocation error of 39 meters. We then exploit technology and deployment constraints to extend the attack to a larger set of IPv6 residential routers by clustering and associating devices with a common penultimate provider router. While we responsibly disclosed our results to several manufacturers and providers, the ossified ecosystem of deployed residential cable and DSL routers suggests that our attack will remain a privacy threat into the foreseeable future.



## **11. Adversarial Texture for Fooling Person Detectors in the Physical World**

cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2203.03373v4)

**Authors**: Zhanhao Hu, Siyuan Huang, Xiaopei Zhu, Fuchun Sun, Bo Zhang, Xiaolin Hu

**Abstracts**: Nowadays, cameras equipped with AI systems can capture and analyze images to detect people automatically. However, the AI system can make mistakes when receiving deliberately designed patterns in the real world, i.e., physical adversarial examples. Prior works have shown that it is possible to print adversarial patches on clothes to evade DNN-based person detectors. However, these adversarial examples could have catastrophic drops in the attack success rate when the viewing angle (i.e., the camera's angle towards the object) changes. To perform a multi-angle attack, we propose Adversarial Texture (AdvTexture). AdvTexture can cover clothes with arbitrary shapes so that people wearing such clothes can hide from person detectors from different viewing angles. We propose a generative method, named Toroidal-Cropping-based Expandable Generative Attack (TC-EGA), to craft AdvTexture with repetitive structures. We printed several pieces of cloth with AdvTexure and then made T-shirts, skirts, and dresses in the physical world. Experiments showed that these clothes could fool person detectors in the physical world.



## **12. An Analytic Framework for Robust Training of Artificial Neural Networks**

cs.LG

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2205.13502v2)

**Authors**: Ramin Barati, Reza Safabakhsh, Mohammad Rahmati

**Abstracts**: The reliability of a learning model is key to the successful deployment of machine learning in various industries. Creating a robust model, particularly one unaffected by adversarial attacks, requires a comprehensive understanding of the adversarial examples phenomenon. However, it is difficult to describe the phenomenon due to the complicated nature of the problems in machine learning. Consequently, many studies investigate the phenomenon by proposing a simplified model of how adversarial examples occur and validate it by predicting some aspect of the phenomenon. While these studies cover many different characteristics of the adversarial examples, they have not reached a holistic approach to the geometric and analytic modeling of the phenomenon. This paper propose a formal framework to study the phenomenon in learning theory and make use of complex analysis and holomorphicity to offer a robust learning rule for artificial neural networks. With the help of complex analysis, we can effortlessly move between geometric and analytic perspectives of the phenomenon and offer further insights on the phenomenon by revealing its connection with harmonic functions. Using our model, we can explain some of the most intriguing characteristics of adversarial examples, including transferability of adversarial examples, and pave the way for novel approaches to mitigate the effects of the phenomenon.



## **13. Revisiting Adversarial Attacks on Graph Neural Networks for Graph Classification**

cs.SI

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2208.06651v1)

**Authors**: Beini Xie, Heng Chang, Xin Wang, Tian Bian, Shiji Zhou, Daixin Wang, Zhiqiang Zhang, Wenwu Zhu

**Abstracts**: Graph neural networks (GNNs) have achieved tremendous success in the task of graph classification and diverse downstream real-world applications. Despite their success, existing approaches are either limited to structure attacks or restricted to local information. This calls for a more general attack framework on graph classification, which faces significant challenges due to the complexity of generating local-node-level adversarial examples using the global-graph-level information. To address this "global-to-local" problem, we present a general framework CAMA to generate adversarial examples by manipulating graph structure and node features in a hierarchical style. Specifically, we make use of Graph Class Activation Mapping and its variant to produce node-level importance corresponding to the graph classification task. Then through a heuristic design of algorithms, we can perform both feature and structure attacks under unnoticeable perturbation budgets with the help of both node-level and subgraph-level importance. Experiments towards attacking four state-of-the-art graph classification models on six real-world benchmarks verify the flexibility and effectiveness of our framework.



## **14. Poison Ink: Robust and Invisible Backdoor Attack**

cs.CR

IEEE Transactions on Image Processing (TIP)

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2108.02488v3)

**Authors**: Jie Zhang, Dongdong Chen, Qidong Huang, Jing Liao, Weiming Zhang, Huamin Feng, Gang Hua, Nenghai Yu

**Abstracts**: Recent research shows deep neural networks are vulnerable to different types of attacks, such as adversarial attack, data poisoning attack and backdoor attack. Among them, backdoor attack is the most cunning one and can occur in almost every stage of deep learning pipeline. Therefore, backdoor attack has attracted lots of interests from both academia and industry. However, most existing backdoor attack methods are either visible or fragile to some effortless pre-processing such as common data transformations. To address these limitations, we propose a robust and invisible backdoor attack called "Poison Ink". Concretely, we first leverage the image structures as target poisoning areas, and fill them with poison ink (information) to generate the trigger pattern. As the image structure can keep its semantic meaning during the data transformation, such trigger pattern is inherently robust to data transformations. Then we leverage a deep injection network to embed such trigger pattern into the cover image to achieve stealthiness. Compared to existing popular backdoor attack methods, Poison Ink outperforms both in stealthiness and robustness. Through extensive experiments, we demonstrate Poison Ink is not only general to different datasets and network architectures, but also flexible for different attack scenarios. Besides, it also has very strong resistance against many state-of-the-art defense techniques.



## **15. MaskBlock: Transferable Adversarial Examples with Bayes Approach**

cs.LG

Under Review

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2208.06538v1)

**Authors**: Mingyuan Fan, Cen Chen, Ximeng Liu, Wenzhong Guo

**Abstracts**: The transferability of adversarial examples (AEs) across diverse models is of critical importance for black-box adversarial attacks, where attackers cannot access the information about black-box models. However, crafted AEs always present poor transferability. In this paper, by regarding the transferability of AEs as generalization ability of the model, we reveal that vanilla black-box attacks craft AEs via solving a maximum likelihood estimation (MLE) problem. For MLE, the results probably are model-specific local optimum when available data is small, i.e., limiting the transferability of AEs. By contrast, we re-formulate crafting transferable AEs as the maximizing a posteriori probability estimation problem, which is an effective approach to boost the generalization of results with limited available data. Because Bayes posterior inference is commonly intractable, a simple yet effective method called MaskBlock is developed to approximately estimate. Moreover, we show that the formulated framework is a generalization version for various attack methods. Extensive experiments illustrate MaskBlock can significantly improve the transferability of crafted adversarial examples by up to about 20%.



## **16. Hide and Seek: on the Stealthiness of Attacks against Deep Learning Systems**

cs.CR

To appear in European Symposium on Research in Computer Security  (ESORICS) 2022

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2205.15944v2)

**Authors**: Zeyan Liu, Fengjun Li, Jingqiang Lin, Zhu Li, Bo Luo

**Abstracts**: With the growing popularity of artificial intelligence and machine learning, a wide spectrum of attacks against deep learning models have been proposed in the literature. Both the evasion attacks and the poisoning attacks attempt to utilize adversarially altered samples to fool the victim model to misclassify the adversarial sample. While such attacks claim to be or are expected to be stealthy, i.e., imperceptible to human eyes, such claims are rarely evaluated. In this paper, we present the first large-scale study on the stealthiness of adversarial samples used in the attacks against deep learning. We have implemented 20 representative adversarial ML attacks on six popular benchmarking datasets. We evaluate the stealthiness of the attack samples using two complementary approaches: (1) a numerical study that adopts 24 metrics for image similarity or quality assessment; and (2) a user study of 3 sets of questionnaires that has collected 20,000+ annotations from 1,000+ responses. Our results show that the majority of the existing attacks introduce nonnegligible perturbations that are not stealthy to human eyes. We further analyze the factors that contribute to attack stealthiness. We further examine the correlation between the numerical analysis and the user studies, and demonstrate that some image quality metrics may provide useful guidance in attack designs, while there is still a significant gap between assessed image quality and visual stealthiness of attacks.



## **17. PRIVEE: A Visual Analytic Workflow for Proactive Privacy Risk Inspection of Open Data**

cs.CR

Accepted for IEEE Symposium on Visualization in Cyber Security, 2022

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06481v1)

**Authors**: Kaustav Bhattacharjee, Akm Islam, Jaideep Vaidya, Aritra Dasgupta

**Abstracts**: Open data sets that contain personal information are susceptible to adversarial attacks even when anonymized. By performing low-cost joins on multiple datasets with shared attributes, malicious users of open data portals might get access to information that violates individuals' privacy. However, open data sets are primarily published using a release-and-forget model, whereby data owners and custodians have little to no cognizance of these privacy risks. We address this critical gap by developing a visual analytic solution that enables data defenders to gain awareness about the disclosure risks in local, joinable data neighborhoods. The solution is derived through a design study with data privacy researchers, where we initially play the role of a red team and engage in an ethical data hacking exercise based on privacy attack scenarios. We use this problem and domain characterization to develop a set of visual analytic interventions as a defense mechanism and realize them in PRIVEE, a visual risk inspection workflow that acts as a proactive monitor for data defenders. PRIVEE uses a combination of risk scores and associated interactive visualizations to let data defenders explore vulnerable joins and interpret risks at multiple levels of data granularity. We demonstrate how PRIVEE can help emulate the attack strategies and diagnose disclosure risks through two case studies with data privacy experts.



## **18. UniNet: A Unified Scene Understanding Network and Exploring Multi-Task Relationships through the Lens of Adversarial Attacks**

cs.CV

Accepted at DeepMTL workshop, ICCV 2021

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2108.04584v2)

**Authors**: Naresh Kumar Gurulingan, Elahe Arani, Bahram Zonooz

**Abstracts**: Scene understanding is crucial for autonomous systems which intend to operate in the real world. Single task vision networks extract information only based on some aspects of the scene. In multi-task learning (MTL), on the other hand, these single tasks are jointly learned, thereby providing an opportunity for tasks to share information and obtain a more comprehensive understanding. To this end, we develop UniNet, a unified scene understanding network that accurately and efficiently infers vital vision tasks including object detection, semantic segmentation, instance segmentation, monocular depth estimation, and monocular instance depth prediction. As these tasks look at different semantic and geometric information, they can either complement or conflict with each other. Therefore, understanding inter-task relationships can provide useful cues to enable complementary information sharing. We evaluate the task relationships in UniNet through the lens of adversarial attacks based on the notion that they can exploit learned biases and task interactions in the neural network. Extensive experiments on the Cityscapes dataset, using untargeted and targeted attacks reveal that semantic tasks strongly interact amongst themselves, and the same holds for geometric tasks. Additionally, we show that the relationship between semantic and geometric tasks is asymmetric and their interaction becomes weaker as we move towards higher-level representations.



## **19. Unifying Gradients to Improve Real-world Robustness for Deep Networks**

stat.ML

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06228v1)

**Authors**: Yingwen Wu, Sizhe Chen, Kun Fang, Xiaolin Huang

**Abstracts**: The wide application of deep neural networks (DNNs) demands an increasing amount of attention to their real-world robustness, i.e., whether a DNN resists black-box adversarial attacks, among them score-based query attacks (SQAs) are the most threatening ones because of their practicalities and effectiveness: the attackers only need dozens of queries on model outputs to seriously hurt a victim network. Defending against SQAs requires a slight but artful variation of outputs due to the service purpose for users, who share the same output information with attackers. In this paper, we propose a real-world defense, called Unifying Gradients (UniG), to unify gradients of different data so that attackers could only probe a much weaker attack direction that is similar for different samples. Since such universal attack perturbations have been validated as less aggressive than the input-specific perturbations, UniG protects real-world DNNs by indicating attackers a twisted and less informative attack direction. To enhance UniG's practical significance in real-world applications, we implement it as a Hadamard product module that is computationally-efficient and readily plugged into any model. According to extensive experiments on 5 SQAs and 4 defense baselines, UniG significantly improves real-world robustness without hurting clean accuracy on CIFAR10 and ImageNet. For instance, UniG maintains a CIFAR-10 model of 77.80% accuracy under 2500-query Square attack while the state-of-the-art adversarially-trained model only has 67.34% on CIFAR10. Simultaneously, UniG greatly surpasses all compared baselines in clean accuracy and the modification degree of outputs. The code would be released.



## **20. Scale-free Photo-realistic Adversarial Pattern Attack**

cs.CV

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06222v1)

**Authors**: Xiangbo Gao, Weicheng Xie, Minmin Liu, Cheng Luo, Qinliang Lin, Linlin Shen, Keerthy Kusumam, Siyang Song

**Abstracts**: Traditional pixel-wise image attack algorithms suffer from poor robustness to defense algorithms, i.e., the attack strength degrades dramatically when defense algorithms are applied. Although Generative Adversarial Networks (GAN) can partially address this problem by synthesizing a more semantically meaningful texture pattern, the main limitation is that existing generators can only generate images of a specific scale. In this paper, we propose a scale-free generation-based attack algorithm that synthesizes semantically meaningful adversarial patterns globally to images with arbitrary scales. Our generative attack approach consistently outperforms the state-of-the-art methods on a wide range of attack settings, i.e. the proposed approach largely degraded the performance of various image classification, object detection, and instance segmentation algorithms under different advanced defense methods.



## **21. A Knowledge Distillation-Based Backdoor Attack in Federated Learning**

cs.LG

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06176v1)

**Authors**: Yifan Wang, Wei Fan, Keke Yang, Naji Alhusaini, Jing Li

**Abstracts**: Federated Learning (FL) is a novel framework of decentralized machine learning. Due to the decentralized feature of FL, it is vulnerable to adversarial attacks in the training procedure, e.g. , backdoor attacks. A backdoor attack aims to inject a backdoor into the machine learning model such that the model will make arbitrarily incorrect behavior on the test sample with some specific backdoor trigger. Even though a range of backdoor attack methods of FL has been introduced, there are also methods defending against them. Many of the defending methods utilize the abnormal characteristics of the models with backdoor or the difference between the models with backdoor and the regular models. To bypass these defenses, we need to reduce the difference and the abnormal characteristics. We find a source of such abnormality is that backdoor attack would directly flip the label of data when poisoning the data. However, current studies of the backdoor attack in FL are not mainly focus on reducing the difference between the models with backdoor and the regular models. In this paper, we propose Adversarial Knowledge Distillation(ADVKD), a method combine knowledge distillation with backdoor attack in FL. With knowledge distillation, we can reduce the abnormal characteristics in model result from the label flipping, thus the model can bypass the defenses. Compared to current methods, we show that ADVKD can not only reach a higher attack success rate, but also successfully bypass the defenses when other methods fails. To further explore the performance of ADVKD, we test how the parameters affect the performance of ADVKD under different scenarios. According to the experiment result, we summarize how to adjust the parameter for better performance under different scenarios. We also use several methods to visualize the effect of different attack and explain the effectiveness of ADVKD.



## **22. A Survey of MulVAL Extensions and Their Attack Scenarios Coverage**

cs.CR

**SubmitDate**: 2022-08-11    [paper-pdf](http://arxiv.org/pdf/2208.05750v1)

**Authors**: David Tayouri, Nick Baum, Asaf Shabtai, Rami Puzis

**Abstracts**: Organizations employ various adversary models in order to assess the risk and potential impact of attacks on their networks. Attack graphs represent vulnerabilities and actions an attacker can take to identify and compromise an organization's assets. Attack graphs facilitate both visual presentation and algorithmic analysis of attack scenarios in the form of attack paths. MulVAL is a generic open-source framework for constructing logical attack graphs, which has been widely used by researchers and practitioners and extended by them with additional attack scenarios. This paper surveys all of the existing MulVAL extensions, and maps all MulVAL interaction rules to MITRE ATT&CK Techniques to estimate their attack scenarios coverage. This survey aligns current MulVAL extensions along unified ontological concepts and highlights the existing gaps. It paves the way for methodical improvement of MulVAL and the comprehensive modeling of the entire landscape of adversarial behaviors captured in MITRE ATT&CK.



## **23. Diverse Generative Adversarial Perturbations on Attention Space for Transferable Adversarial Attacks**

cs.CV

ICIP 2022

**SubmitDate**: 2022-08-11    [paper-pdf](http://arxiv.org/pdf/2208.05650v1)

**Authors**: Woo Jae Kim, Seunghoon Hong, Sung-Eui Yoon

**Abstracts**: Adversarial attacks with improved transferability - the ability of an adversarial example crafted on a known model to also fool unknown models - have recently received much attention due to their practicality. Nevertheless, existing transferable attacks craft perturbations in a deterministic manner and often fail to fully explore the loss surface, thus falling into a poor local optimum and suffering from low transferability. To solve this problem, we propose Attentive-Diversity Attack (ADA), which disrupts diverse salient features in a stochastic manner to improve transferability. Primarily, we perturb the image attention to disrupt universal features shared by different models. Then, to effectively avoid poor local optima, we disrupt these features in a stochastic manner and explore the search space of transferable perturbations more exhaustively. More specifically, we use a generator to produce adversarial perturbations that each disturbs features in different ways depending on an input latent code. Extensive experimental evaluations demonstrate the effectiveness of our method, outperforming the transferability of state-of-the-art methods. Codes are available at https://github.com/wkim97/ADA.



## **24. Controlled Quantum Teleportation in the Presence of an Adversary**

quant-ph

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05554v1)

**Authors**: Sayan Gangopadhyay, Tiejun Wang, Atefeh Mashatan, Shohini Ghose

**Abstracts**: We present a device independent analysis of controlled quantum teleportation where the receiver is not trusted. We show that the notion of genuine tripartite nonlocality allows us to certify control power in such a scenario. By considering a specific adversarial attack strategy on a device characterized by depolarizing noise, we find that control power is a monotonically increasing function of genuine tripartite nonlocality. These results are relevant for building practical quantum communication networks and also shed light on the role of nonlocality in multipartite quantum information processing.



## **25. Pikachu: Securing PoS Blockchains from Long-Range Attacks by Checkpointing into Bitcoin PoW using Taproot**

cs.CR

To appear at ConsensusDay 22 (ACM CCS 2022 Workshop)

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05408v1)

**Authors**: Sarah Azouvi, Marko Vukolić

**Abstracts**: Blockchain systems based on a reusable resource, such as proof-of-stake (PoS), provide weaker security guarantees than those based on proof-of-work. Specifically, they are vulnerable to long-range attacks, where an adversary can corrupt prior participants in order to rewrite the full history of the chain. To prevent this attack on a PoS chain, we propose a protocol that checkpoints the state of the PoS chain to a proof-of-work blockchain such as Bitcoin. Our checkpointing protocol hence does not rely on any central authority. Our work uses Schnorr signatures and leverages Bitcoin recent Taproot upgrade, allowing us to create a checkpointing transaction of constant size. We argue for the security of our protocol and present an open-source implementation that was tested on the Bitcoin testnet.



## **26. StratDef: a strategic defense against adversarial attacks in malware detection**

cs.LG

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2202.07568v2)

**Authors**: Aqib Rashid, Jose Such

**Abstracts**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system tailored for the malware detection domain based on a moving target defense approach. We overcome challenges related to the systematic construction, selection and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker, whilst minimizing critical aspects in the adversarial ML domain like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, from the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.



## **27. Reducing Exploitability with Population Based Training**

cs.LG

Presented at New Frontiers in Adversarial Machine Learning Workshop,  ICML 2022

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05083v1)

**Authors**: Pavel Czempin, Adam Gleave

**Abstracts**: Self-play reinforcement learning has achieved state-of-the-art, and often superhuman, performance in a variety of zero-sum games. Yet prior work has found that policies that are highly capable against regular opponents can fail catastrophically against adversarial policies: an opponent trained explicitly against the victim. Prior defenses using adversarial training were able to make the victim robust to a specific adversary, but the victim remained vulnerable to new ones. We conjecture this limitation was due to insufficient diversity of adversaries seen during training. We propose a defense using population based training to pit the victim against a diverse set of opponents. We evaluate this defense's robustness against new adversaries in two low-dimensional environments. Our defense increases robustness against adversaries, as measured by number of attacker training timesteps to exploit the victim. Furthermore, we show that robustness is correlated with the size of the opponent population.



## **28. Adversarial Machine Learning-Based Anticipation of Threats Against Vehicle-to-Microgrid Services**

cs.CR

IEEE Global Communications Conference (Globecom), 2022, 6 pages, 2  Figures, 4 Tables

**SubmitDate**: 2022-08-09    [paper-pdf](http://arxiv.org/pdf/2208.05073v1)

**Authors**: Ahmed Omara, Burak Kantarci

**Abstracts**: In this paper, we study the expanding attack surface of Adversarial Machine Learning (AML) and the potential attacks against Vehicle-to-Microgrid (V2M) services. We present an anticipatory study of a multi-stage gray-box attack that can achieve a comparable result to a white-box attack. Adversaries aim to deceive the targeted Machine Learning (ML) classifier at the network edge to misclassify the incoming energy requests from microgrids. With an inference attack, an adversary can collect real-time data from the communication between smart microgrids and a 5G gNodeB to train a surrogate (i.e., shadow) model of the targeted classifier at the edge. To anticipate the associated impact of an adversary's capability to collect real-time data instances, we study five different cases, each representing different amounts of real-time data instances collected by an adversary. Out of six ML models trained on the complete dataset, K-Nearest Neighbour (K-NN) is selected as the surrogate model, and through simulations, we demonstrate that the multi-stage gray-box attack is able to mislead the ML classifier and cause an Evasion Increase Rate (EIR) up to 73.2% using 40% less data than what a white-box attack needs to achieve a similar EIR.



## **29. Get your Foes Fooled: Proximal Gradient Split Learning for Defense against Model Inversion Attacks on IoMT data**

cs.CR

10 pages, 5 figures, 2 tables

**SubmitDate**: 2022-08-09    [paper-pdf](http://arxiv.org/pdf/2201.04569v3)

**Authors**: Sunder Ali Khowaja, Ik Hyun Lee, Kapal Dev, Muhammad Aslam Jarwar, Nawab Muhammad Faseeh Qureshi

**Abstracts**: The past decade has seen a rapid adoption of Artificial Intelligence (AI), specifically the deep learning networks, in Internet of Medical Things (IoMT) ecosystem. However, it has been shown recently that the deep learning networks can be exploited by adversarial attacks that not only make IoMT vulnerable to the data theft but also to the manipulation of medical diagnosis. The existing studies consider adding noise to the raw IoMT data or model parameters which not only reduces the overall performance concerning medical inferences but also is ineffective to the likes of deep leakage from gradients method. In this work, we propose proximal gradient split learning (PSGL) method for defense against the model inversion attacks. The proposed method intentionally attacks the IoMT data when undergoing the deep neural network training process at client side. We propose the use of proximal gradient method to recover gradient maps and a decision-level fusion strategy to improve the recognition performance. Extensive analysis show that the PGSL not only provides effective defense mechanism against the model inversion attacks but also helps in improving the recognition performance on publicly available datasets. We report 14.0$\%$, 17.9$\%$, and 36.9$\%$ gains in accuracy over reconstructed and adversarial attacked images, respectively.



## **30. Bayesian Pseudo Labels: Expectation Maximization for Robust and Efficient Semi-Supervised Segmentation**

cs.CV

MICCAI 2022 (Early accept, Student Travel Award)

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2208.04435v1)

**Authors**: Mou-Cheng Xu, Yukun Zhou, Chen Jin, Marius de Groot, Daniel C. Alexander, Neil P. Oxtoby, Yipeng Hu, Joseph Jacob

**Abstracts**: This paper concerns pseudo labelling in segmentation. Our contribution is fourfold. Firstly, we present a new formulation of pseudo-labelling as an Expectation-Maximization (EM) algorithm for clear statistical interpretation. Secondly, we propose a semi-supervised medical image segmentation method purely based on the original pseudo labelling, namely SegPL. We demonstrate SegPL is a competitive approach against state-of-the-art consistency regularisation based methods on semi-supervised segmentation on a 2D multi-class MRI brain tumour segmentation task and a 3D binary CT lung vessel segmentation task. The simplicity of SegPL allows less computational cost comparing to prior methods. Thirdly, we demonstrate that the effectiveness of SegPL may originate from its robustness against out-of-distribution noises and adversarial attacks. Lastly, under the EM framework, we introduce a probabilistic generalisation of SegPL via variational inference, which learns a dynamic threshold for pseudo labelling during the training. We show that SegPL with variational inference can perform uncertainty estimation on par with the gold-standard method Deep Ensemble.



## **31. Can collaborative learning be private, robust and scalable?**

cs.LG

Accepted at MICCAI DeCaF 2022

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2205.02652v2)

**Authors**: Dmitrii Usynin, Helena Klause, Johannes C. Paetzold, Daniel Rueckert, Georgios Kaissis

**Abstracts**: In federated learning for medical image analysis, the safety of the learning protocol is paramount. Such settings can often be compromised by adversaries that target either the private data used by the federation or the integrity of the model itself. This requires the medical imaging community to develop mechanisms to train collaborative models that are private and robust against adversarial data. In response to these challenges, we propose a practical open-source framework to study the effectiveness of combining differential privacy, model compression and adversarial training to improve the robustness of models against adversarial samples under train- and inference-time attacks. Using our framework, we achieve competitive model performance, a significant reduction in model's size and an improved empirical adversarial robustness without a severe performance degradation, critical in medical image analysis.



## **32. Sparse Adversarial Attack in Multi-agent Reinforcement Learning**

cs.AI

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2205.09362v2)

**Authors**: Yizheng Hu, Zhihua Zhang

**Abstracts**: Cooperative multi-agent reinforcement learning (cMARL) has many real applications, but the policy trained by existing cMARL algorithms is not robust enough when deployed. There exist also many methods about adversarial attacks on the RL system, which implies that the RL system can suffer from adversarial attacks, but most of them focused on single agent RL. In this paper, we propose a \textit{sparse adversarial attack} on cMARL systems. We use (MA)RL with regularization to train the attack policy. Our experiments show that the policy trained by the current cMARL algorithm can obtain poor performance when only one or a few agents in the team (e.g., 1 of 8 or 5 of 25) were attacked at a few timesteps (e.g., attack 3 of total 40 timesteps).



## **33. Adversarial Pixel Restoration as a Pretext Task for Transferable Perturbations**

cs.CV

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2207.08803v2)

**Authors**: Hashmat Shadab Malik, Shahina K Kunhimon, Muzammal Naseer, Salman Khan, Fahad Shahbaz Khan

**Abstracts**: Transferable adversarial attacks optimize adversaries from a pretrained surrogate model and known label space to fool the unknown black-box models. Therefore, these attacks are restricted by the availability of an effective surrogate model. In this work, we relax this assumption and propose Adversarial Pixel Restoration as a self-supervised alternative to train an effective surrogate model from scratch under the condition of no labels and few data samples. Our training approach is based on a min-max scheme which reduces overfitting via an adversarial objective and thus optimizes for a more generalizable surrogate model. Our proposed attack is complimentary to the adversarial pixel restoration and is independent of any task specific objective as it can be launched in a self-supervised manner. We successfully demonstrate the adversarial transferability of our approach to Vision Transformers as well as Convolutional Neural Networks for the tasks of classification, object detection, and video segmentation. Our training approach improves the transferability of the baseline unsupervised training method by 16.4% on ImageNet val. set. Our codes & pre-trained surrogate models are available at: https://github.com/HashmatShadab/APR



## **34. Adversarial robustness of $β-$VAE through the lens of local geometry**

cs.LG

The 2022 ICML Workshop on New Frontiers in Adversarial Machine  Learning

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2208.03923v1)

**Authors**: Asif Khan, Amos Storkey

**Abstracts**: Variational autoencoders (VAEs) are susceptible to adversarial attacks. An adversary can find a small perturbation in the input sample to change its latent encoding non-smoothly, thereby compromising the reconstruction. A known reason for such vulnerability is the latent space distortions arising from a mismatch between approximated latent posterior and a prior distribution. Consequently, a slight change in the inputs leads to a significant change in the latent space encodings. This paper demonstrates that the sensitivity around a data point is due to a directional bias of a stochastic pullback metric tensor induced by the encoder network. The pullback metric tensor measures the infinitesimal volume change from input to latent space. Thus, it can be viewed as a lens to analyse the effect of small changes in the input leading to distortions in the latent space. We propose robustness evaluation scores using the eigenspectrum of a pullback metric. Moreover, we empirically show that the scores correlate with the robustness parameter $\beta$ of the $\beta-$VAE.



## **35. Adversarial Fine-tuning for Backdoor Defense: Connecting Backdoor Attacks to Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2202.06312v3)

**Authors**: Bingxu Mu, Zhenxing Niu, Le Wang, Xue Wang, Rong Jin, Gang Hua

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to both backdoor attacks as well as adversarial attacks. In the literature, these two types of attacks are commonly treated as distinct problems and solved separately, since they belong to training-time and inference-time attacks respectively. However, in this paper we find an intriguing connection between them: for a model planted with backdoors, we observe that its adversarial examples have similar behaviors as its triggered samples, i.e., both activate the same subset of DNN neurons. It indicates that planting a backdoor into a model will significantly affect the model's adversarial examples. Based on this observations, we design a new Adversarial Fine-Tuning (AFT) algorithm to defend against backdoor attacks. We empirically show that, against 5 state-of-the-art backdoor attacks, our AFT can effectively erase the backdoor triggers without obvious performance degradation on clean samples and significantly outperforms existing defense methods.



## **36. Privacy Against Inference Attacks in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2207.11788v2)

**Authors**: Borzoo Rassouli, Morteza Varasteh, Deniz Gunduz

**Abstracts**: Vertical federated learning is considered, where an active party, having access to true class labels, wishes to build a classification model by utilizing more features from a passive party, which has no access to the labels, to improve the model accuracy. In the prediction phase, with logistic regression as the classification model, several inference attack techniques are proposed that the adversary, i.e., the active party, can employ to reconstruct the passive party's features, regarded as sensitive information. These attacks, which are mainly based on a classical notion of the center of a set, i.e., the Chebyshev center, are shown to be superior to those proposed in the literature. Moreover, several theoretical performance guarantees are provided for the aforementioned attacks. Subsequently, we consider the minimum amount of information that the adversary needs to fully reconstruct the passive party's features. In particular, it is shown that when the passive party holds one feature, and the adversary is only aware of the signs of the parameters involved, it can perfectly reconstruct that feature when the number of predictions is large enough. Next, as a defense mechanism, a privacy-preserving scheme is proposed that worsen the adversary's reconstruction attacks, while preserving the full benefits that VFL brings to the active party. Finally, experimental results demonstrate the effectiveness of the proposed attacks and the privacy-preserving scheme.



## **37. Garbled EDA: Privacy Preserving Electronic Design Automation**

cs.CR

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2208.03822v1)

**Authors**: Mohammad Hashemi, Steffi Roy, Fatemeh Ganji, Domenic Forte

**Abstracts**: The complexity of modern integrated circuits (ICs) necessitates collaboration between multiple distrusting parties, including thirdparty intellectual property (3PIP) vendors, design houses, CAD/EDA tool vendors, and foundries, which jeopardizes confidentiality and integrity of each party's IP. IP protection standards and the existing techniques proposed by researchers are ad hoc and vulnerable to numerous structural, functional, and/or side-channel attacks. Our framework, Garbled EDA, proposes an alternative direction through formulating the problem in a secure multi-party computation setting, where the privacy of IPs, CAD tools, and process design kits (PDKs) is maintained. As a proof-of-concept, Garbled EDA is evaluated in the context of simulation, where multiple IP description formats (Verilog, C, S) are supported. Our results demonstrate a reasonable logical-resource cost and negligible memory overhead. To further reduce the overhead, we present another efficient implementation methodology, feasible when the resource utilization is a bottleneck, but the communication between two parties is not restricted. Interestingly, this implementation is private and secure even in the presence of malicious adversaries attempting to, e.g., gain access to PDKs or in-house IPs of the CAD tool providers.



## **38. Federated Adversarial Learning: A Framework with Convergence Analysis**

cs.LG

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2208.03635v1)

**Authors**: Xiaoxiao Li, Zhao Song, Jiaming Yang

**Abstracts**: Federated learning (FL) is a trending training paradigm to utilize decentralized training data. FL allows clients to update model parameters locally for several epochs, then share them to a global model for aggregation. This training paradigm with multi-local step updating before aggregation exposes unique vulnerabilities to adversarial attacks. Adversarial training is a popular and effective method to improve the robustness of networks against adversaries. In this work, we formulate a general form of federated adversarial learning (FAL) that is adapted from adversarial learning in the centralized setting. On the client side of FL training, FAL has an inner loop to generate adversarial samples for adversarial training and an outer loop to update local model parameters. On the server side, FAL aggregates local model updates and broadcast the aggregated model. We design a global robust training loss and formulate FAL training as a min-max optimization problem. Unlike the convergence analysis in classical centralized training that relies on the gradient direction, it is significantly harder to analyze the convergence in FAL for three reasons: 1) the complexity of min-max optimization, 2) model not updating in the gradient direction due to the multi-local updates on the client-side before aggregation and 3) inter-client heterogeneity. We address these challenges by using appropriate gradient approximation and coupling techniques and present the convergence analysis in the over-parameterized regime. Our main result theoretically shows that the minimum loss under our algorithm can converge to $\epsilon$ small with chosen learning rate and communication rounds. It is noteworthy that our analysis is feasible for non-IID clients.



## **39. Blackbox Attacks via Surrogate Ensemble Search**

cs.LG

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2208.03610v1)

**Authors**: Zikui Cai, Chengyu Song, Srikanth Krishnamurthy, Amit Roy-Chowdhury, M. Salman Asif

**Abstracts**: Blackbox adversarial attacks can be categorized into transfer- and query-based attacks. Transfer methods do not require any feedback from the victim model, but provide lower success rates compared to query-based methods. Query attacks often require a large number of queries for success. To achieve the best of both approaches, recent efforts have tried to combine them, but still require hundreds of queries to achieve high success rates (especially for targeted attacks). In this paper, we propose a novel method for blackbox attacks via surrogate ensemble search (BASES) that can generate highly successful blackbox attacks using an extremely small number of queries. We first define a perturbation machine that generates a perturbed image by minimizing a weighted loss function over a fixed set of surrogate models. To generate an attack for a given victim model, we search over the weights in the loss function using queries generated by the perturbation machine. Since the dimension of the search space is small (same as the number of surrogate models), the search requires a small number of queries. We demonstrate that our proposed method achieves better success rate with at least 30x fewer queries compared to state-of-the-art methods on different image classifiers trained with ImageNet (including VGG-19, DenseNet-121, and ResNext-50). In particular, our method requires as few as 3 queries per image (on average) to achieve more than a 90% success rate for targeted attacks and 1-2 queries per image for over a 99% success rate for non-targeted attacks. Our method is also effective on Google Cloud Vision API and achieved a 91% non-targeted attack success rate with 2.9 queries per image. We also show that the perturbations generated by our proposed method are highly transferable and can be adopted for hard-label blackbox attacks.



## **40. Revisiting Gaussian Neurons for Online Clustering with Unknown Number of Clusters**

cs.LG

Reviewed at  https://openreview.net/forum?id=h05RLBNweX&referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR)

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2205.00920v2)

**Authors**: Ole Christian Eidheim

**Abstracts**: Despite the recent success of artificial neural networks, more biologically plausible learning methods may be needed to resolve the weaknesses of backpropagation trained models such as catastrophic forgetting and adversarial attacks. Although these weaknesses are not specifically addressed, a novel local learning rule is presented that performs online clustering with an upper limit on the number of clusters to be found rather than a fixed cluster count. Instead of using orthogonal weight or output activation constraints, activation sparsity is achieved by mutual repulsion of lateral Gaussian neurons ensuring that multiple neuron centers cannot occupy the same location in the input domain. An update method is also presented for adjusting the widths of the Gaussian neurons in cases where the data samples can be represented by means and variances. The algorithms were applied on the MNIST and CIFAR-10 datasets to create filters capturing the input patterns of pixel patches of various sizes. The experimental results demonstrate stability in the learned parameters across a large number of training samples.



## **41. On the Fundamental Limits of Formally (Dis)Proving Robustness in Proof-of-Learning**

cs.LG

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2208.03567v1)

**Authors**: Congyu Fang, Hengrui Jia, Anvith Thudi, Mohammad Yaghini, Christopher A. Choquette-Choo, Natalie Dullerud, Varun Chandrasekaran, Nicolas Papernot

**Abstracts**: Proof-of-learning (PoL) proposes a model owner use machine learning training checkpoints to establish a proof of having expended the necessary compute for training. The authors of PoL forego cryptographic approaches and trade rigorous security guarantees for scalability to deep learning by being applicable to stochastic gradient descent and adaptive variants. This lack of formal analysis leaves the possibility that an attacker may be able to spoof a proof for a model they did not train.   We contribute a formal analysis of why the PoL protocol cannot be formally (dis)proven to be robust against spoofing adversaries. To do so, we disentangle the two roles of proof verification in PoL: (a) efficiently determining if a proof is a valid gradient descent trajectory, and (b) establishing precedence by making it more expensive to craft a proof after training completes (i.e., spoofing). We show that efficient verification results in a tradeoff between accepting legitimate proofs and rejecting invalid proofs because deep learning necessarily involves noise. Without a precise analytical model for how this noise affects training, we cannot formally guarantee if a PoL verification algorithm is robust. Then, we demonstrate that establishing precedence robustly also reduces to an open problem in learning theory: spoofing a PoL post hoc training is akin to finding different trajectories with the same endpoint in non-convex learning. Yet, we do not rigorously know if priori knowledge of the final model weights helps discover such trajectories.   We conclude that, until the aforementioned open problems are addressed, relying more heavily on cryptography is likely needed to formulate a new class of PoL protocols with formal robustness guarantees. In particular, this will help with establishing precedence. As a by-product of insights from our analysis, we also demonstrate two novel attacks against PoL.



## **42. Preventing or Mitigating Adversarial Supply Chain Attacks; a legal analysis**

cs.CY

23 pages

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2208.03466v1)

**Authors**: Kaspar Rosager Ludvigsen, Shishir Nagaraja, Angela Daly

**Abstracts**: The world is currently strongly connected through both the internet at large, but also the very supply chains which provide everything from food to infrastructure and technology. The supply chains are themselves vulnerable to adversarial attacks, both in a digital and physical sense, which can disrupt or at worst destroy them. In this paper, we take a look at two examples of such successful attacks and consider what their consequences may be going forward, and analyse how EU and national law can prevent these attacks or otherwise punish companies which do not try to mitigate them at all possible costs. We find that the current types of national regulation are not technology specific enough, and cannot force or otherwise mandate the correct parties who could play the biggest role in preventing supply chain attacks to do everything in their power to mitigate them. But, current EU law is on the right path, and further vigilance may be what is necessary to consider these large threats, as national law tends to fail at properly regulating companies when it comes to cybersecurity.



## **43. Searching for the Essence of Adversarial Perturbations**

cs.LG

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2205.15357v2)

**Authors**: Dennis Y. Menn, Hung-yi Lee

**Abstracts**: Neural networks have achieved the state-of-the-art performance in various machine learning fields, yet the incorporation of malicious perturbations with input data (adversarial example) is shown to fool neural networks' predictions. This would lead to potential risks for real-world applications such as endangering autonomous driving and messing up text identification. To mitigate such risks, an understanding of how adversarial examples operate is critical, which however remains unresolved. Here we demonstrate that adversarial perturbations contain human-recognizable information, which is the key conspirator responsible for a neural network's erroneous prediction, in contrast to a widely discussed argument that human-imperceptible information plays the critical role in fooling a network. This concept of human-recognizable information allows us to explain key features related to adversarial perturbations, including the existence of adversarial examples, the transferability among different neural networks, and the increased neural network interpretability for adversarial training. Two unique properties in adversarial perturbations that fool neural networks are uncovered: masking and generation. A special class, the complementary class, is identified when neural networks classify input images. The human-recognizable information contained in adversarial perturbations allows researchers to gain insight on the working principles of neural networks and may lead to develop techniques that detect/defense adversarial attacks.



## **44. Success of Uncertainty-Aware Deep Models Depends on Data Manifold Geometry**

cs.LG

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2208.01705v2)

**Authors**: Mark Penrod, Harrison Termotto, Varshini Reddy, Jiayu Yao, Finale Doshi-Velez, Weiwei Pan

**Abstracts**: For responsible decision making in safety-critical settings, machine learning models must effectively detect and process edge-case data. Although existing works show that predictive uncertainty is useful for these tasks, it is not evident from literature which uncertainty-aware models are best suited for a given dataset. Thus, we compare six uncertainty-aware deep learning models on a set of edge-case tasks: robustness to adversarial attacks as well as out-of-distribution and adversarial detection. We find that the geometry of the data sub-manifold is an important factor in determining the success of various models. Our finding suggests an interesting direction in the study of uncertainty-aware deep learning models.



## **45. Attacking Adversarial Defences by Smoothing the Loss Landscape**

cs.LG

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2208.00862v2)

**Authors**: Panagiotis Eustratiadis, Henry Gouk, Da Li, Timothy Hospedales

**Abstracts**: This paper investigates a family of methods for defending against adversarial attacks that owe part of their success to creating a noisy, discontinuous, or otherwise rugged loss landscape that adversaries find difficult to navigate. A common, but not universal, way to achieve this effect is via the use of stochastic neural networks. We show that this is a form of gradient obfuscation, and propose a general extension to gradient-based adversaries based on the Weierstrass transform, which smooths the surface of the loss function and provides more reliable gradient estimates. We further show that the same principle can strengthen gradient-free adversaries. We demonstrate the efficacy of our loss-smoothing method against both stochastic and non-stochastic adversarial defences that exhibit robustness due to this type of obfuscation. Furthermore, we provide analysis of how it interacts with Expectation over Transformation; a popular gradient-sampling method currently used to attack stochastic defences.



## **46. Adversarial Robustness of MR Image Reconstruction under Realistic Perturbations**

eess.IV

Accepted at the MICCAI-2022 workshop: Machine Learning for Medical  Image Reconstruction

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2208.03161v1)

**Authors**: Jan Nikolas Morshuis, Sergios Gatidis, Matthias Hein, Christian F. Baumgartner

**Abstracts**: Deep Learning (DL) methods have shown promising results for solving ill-posed inverse problems such as MR image reconstruction from undersampled $k$-space data. However, these approaches currently have no guarantees for reconstruction quality and the reliability of such algorithms is only poorly understood. Adversarial attacks offer a valuable tool to understand possible failure modes and worst case performance of DL-based reconstruction algorithms. In this paper we describe adversarial attacks on multi-coil $k$-space measurements and evaluate them on the recently proposed E2E-VarNet and a simpler UNet-based model. In contrast to prior work, the attacks are targeted to specifically alter diagnostically relevant regions. Using two realistic attack models (adversarial $k$-space noise and adversarial rotations) we are able to show that current state-of-the-art DL-based reconstruction algorithms are indeed sensitive to such perturbations to a degree where relevant diagnostic information may be lost. Surprisingly, in our experiments the UNet and the more sophisticated E2E-VarNet were similarly sensitive to such attacks. Our findings add further to the evidence that caution must be exercised as DL-based methods move closer to clinical practice.



## **47. A Systematic Survey of Attack Detection and Prevention in Connected and Autonomous Vehicles**

cs.CR

This article is published in the Vehicular Communications journal

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2203.14965v2)

**Authors**: Trupil Limbasiya, Ko Zheng Teng, Sudipta Chattopadhyay, Jianying Zhou

**Abstracts**: The number of Connected and Autonomous Vehicles (CAVs) is increasing rapidly in various smart transportation services and applications, considering many benefits to society, people, and the environment. Several research surveys for CAVs were conducted by primarily focusing on various security threats and vulnerabilities in the domain of CAVs to classify different types of attacks, impacts of attacks, attack features, cyber-risk, defense methodologies against attacks, and safety standards. However, the importance of attack detection and prevention approaches for CAVs has not been discussed extensively in the state-of-the-art surveys, and there is a clear gap in the existing literature on such methodologies to detect new and conventional threats and protect the CAV systems from unexpected hazards on the road. Some surveys have a limited discussion on Attacks Detection and Prevention Systems (ADPS), but such surveys provide only partial coverage of different types of ADPS for CAVs. Furthermore, there is a scope for discussing security, privacy, and efficiency challenges in ADPS that can give an overview of important security and performance attributes.   This survey paper, therefore, presents the significance of CAVs in the market, potential challenges in CAVs, key requirements of essential security and privacy properties, various capabilities of adversaries, possible attacks in CAVs, and performance evaluation parameters for ADPS. An extensive analysis is discussed of different ADPS categories for CAVs and state-of-the-art research works based on each ADPS category that gives the latest findings in this research domain. This survey also discusses crucial and open security research problems that are required to be focused on the secure deployment of CAVs in the market.



## **48. Differentially Private Counterfactuals via Functional Mechanism**

cs.LG

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02878v1)

**Authors**: Fan Yang, Qizhang Feng, Kaixiong Zhou, Jiahao Chen, Xia Hu

**Abstracts**: Counterfactual, serving as one emerging type of model explanation, has attracted tons of attentions recently from both industry and academia. Different from the conventional feature-based explanations (e.g., attributions), counterfactuals are a series of hypothetical samples which can flip model decisions with minimal perturbations on queries. Given valid counterfactuals, humans are capable of reasoning under ``what-if'' circumstances, so as to better understand the model decision boundaries. However, releasing counterfactuals could be detrimental, since it may unintentionally leak sensitive information to adversaries, which brings about higher risks on both model security and data privacy. To bridge the gap, in this paper, we propose a novel framework to generate differentially private counterfactual (DPC) without touching the deployed model or explanation set, where noises are injected for protection while maintaining the explanation roles of counterfactual. In particular, we train an autoencoder with the functional mechanism to construct noisy class prototypes, and then derive the DPC from the latent prototypes based on the post-processing immunity of differential privacy. Further evaluations demonstrate the effectiveness of the proposed framework, showing that DPC can successfully relieve the risks on both extraction and inference attacks.



## **49. Self-Ensembling Vision Transformer (SEViT) for Robust Medical Image Classification**

cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02851v1)

**Authors**: Faris Almalik, Mohammad Yaqub, Karthik Nandakumar

**Abstracts**: Vision Transformers (ViT) are competing to replace Convolutional Neural Networks (CNN) for various computer vision tasks in medical imaging such as classification and segmentation. While the vulnerability of CNNs to adversarial attacks is a well-known problem, recent works have shown that ViTs are also susceptible to such attacks and suffer significant performance degradation under attack. The vulnerability of ViTs to carefully engineered adversarial samples raises serious concerns about their safety in clinical settings. In this paper, we propose a novel self-ensembling method to enhance the robustness of ViT in the presence of adversarial attacks. The proposed Self-Ensembling Vision Transformer (SEViT) leverages the fact that feature representations learned by initial blocks of a ViT are relatively unaffected by adversarial perturbations. Learning multiple classifiers based on these intermediate feature representations and combining these predictions with that of the final ViT classifier can provide robustness against adversarial attacks. Measuring the consistency between the various predictions can also help detect adversarial samples. Experiments on two modalities (chest X-ray and fundoscopy) demonstrate the efficacy of SEViT architecture to defend against various adversarial attacks in the gray-box (attacker has full knowledge of the target model, but not the defense mechanism) setting. Code: https://github.com/faresmalik/SEViT



## **50. Adversarial Attacks on Image Generation With Made-Up Words**

cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.04135v1)

**Authors**: Raphaël Millière

**Abstracts**: Text-guided image generation models can be prompted to generate images using nonce words adversarially designed to robustly evoke specific visual concepts. Two approaches for such generation are introduced: macaronic prompting, which involves designing cryptic hybrid words by concatenating subword units from different languages; and evocative prompting, which involves designing nonce words whose broad morphological features are similar enough to that of existing words to trigger robust visual associations. The two methods can also be combined to generate images associated with more specific visual concepts. The implications of these techniques for the circumvention of existing approaches to content moderation, and particularly the generation of offensive or harmful images, are discussed.



