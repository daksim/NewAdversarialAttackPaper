# Latest Adversarial Attack Papers
**update at 2022-04-14 13:30:26**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Overparameterized Linear Regression under Adversarial Attacks**

stat.ML

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2204.06274v1)

**Authors**: Antônio H. Ribeiro, Thomas B. Schön

**Abstracts**: As machine learning models start to be used in critical applications, their vulnerabilities and brittleness become a pressing concern. Adversarial attacks are a popular framework for studying these vulnerabilities. In this work, we study the error of linear regression in the face of adversarial attacks. We provide bounds of the error in terms of the traditional risk and the parameter norm and show how these bounds can be leveraged and make it possible to use analysis from non-adversarial setups to study the adversarial risk. The usefulness of these results is illustrated by shedding light on whether or not overparameterized linear models can be adversarially robust. We show that adding features to linear models might be either a source of additional robustness or brittleness. We show that these differences appear due to scaling and how the $\ell_1$ and $\ell_2$ norms of random projections concentrate. We also show how the reformulation we propose allows for solving adversarial training as a convex optimization problem. This is then used as a tool to study how adversarial training and other regularization methods might affect the robustness of the estimated models.



## **2. Towards A Critical Evaluation of Robustness for Deep Learning Backdoor Countermeasures**

cs.CR

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2204.06273v1)

**Authors**: Huming Qiu, Hua Ma, Zhi Zhang, Alsharif Abuadbba, Wei Kang, Anmin Fu, Yansong Gao

**Abstracts**: Since Deep Learning (DL) backdoor attacks have been revealed as one of the most insidious adversarial attacks, a number of countermeasures have been developed with certain assumptions defined in their respective threat models. However, the robustness of these countermeasures is inadvertently ignored, which can introduce severe consequences, e.g., a countermeasure can be misused and result in a false implication of backdoor detection.   For the first time, we critically examine the robustness of existing backdoor countermeasures with an initial focus on three influential model-inspection ones that are Neural Cleanse (S&P'19), ABS (CCS'19), and MNTD (S&P'21). Although the three countermeasures claim that they work well under their respective threat models, they have inherent unexplored non-robust cases depending on factors such as given tasks, model architectures, datasets, and defense hyper-parameter, which are \textit{not even rooted from delicate adaptive attacks}. We demonstrate how to trivially bypass them aligned with their respective threat models by simply varying aforementioned factors. Particularly, for each defense, formal proofs or empirical studies are used to reveal its two non-robust cases where it is not as robust as it claims or expects, especially the recent MNTD. This work highlights the necessity of thoroughly evaluating the robustness of backdoor countermeasures to avoid their misleading security implications in unknown non-robust cases.



## **3. Towards Practical Robustness Analysis for DNNs based on PAC-Model Learning**

cs.LG

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2101.10102v2)

**Authors**: Renjue Li, Pengfei Yang, Cheng-Chao Huang, Youcheng Sun, Bai Xue, Lijun Zhang

**Abstracts**: To analyse local robustness properties of deep neural networks (DNNs), we present a practical framework from a model learning perspective. Based on black-box model learning with scenario optimisation, we abstract the local behaviour of a DNN via an affine model with the probably approximately correct (PAC) guarantee. From the learned model, we can infer the corresponding PAC-model robustness property. The innovation of our work is the integration of model learning into PAC robustness analysis: that is, we construct a PAC guarantee on the model level instead of sample distribution, which induces a more faithful and accurate robustness evaluation. This is in contrast to existing statistical methods without model learning. We implement our method in a prototypical tool named DeepPAC. As a black-box method, DeepPAC is scalable and efficient, especially when DNNs have complex structures or high-dimensional inputs. We extensively evaluate DeepPAC, with 4 baselines (using formal verification, statistical methods, testing and adversarial attack) and 20 DNN models across 3 datasets, including MNIST, CIFAR-10, and ImageNet. It is shown that DeepPAC outperforms the state-of-the-art statistical method PROVERO, and it achieves more practical robustness analysis than the formal verification tool ERAN. Also, its results are consistent with existing DNN testing work like DeepGini.



## **4. Stealing Malware Classifiers and AVs at Low False Positive Conditions**

cs.CR

12 pages, 8 figures, 6 tables. Under review

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2204.06241v1)

**Authors**: Maria Rigaki, Sebastian Garcia

**Abstracts**: Model stealing attacks have been successfully used in many machine learning domains, but there is little understanding of how these attacks work in the malware detection domain. Malware detection and, in general, security domains have very strong requirements of low false positive rates (FPR). However, these requirements are not the primary focus of the existing model stealing literature. Stealing attacks create surrogate models that perform similarly to a target model using a limited amount of queries to the target. The first stage of this study is the evaluation of active learning model stealing attacks against publicly available stand-alone machine learning malware classifiers and antivirus products (AVs). We propose a new neural network architecture for surrogate models that outperforms the existing state of the art on low FPR conditions. The surrogates were evaluated on their agreement with the targeted models. Good surrogates of the stand-alone classifiers were created with up to 99% agreement with the target models, using less than 4% of the original training dataset size. Good AV surrogates were also possible to train, but with a lower agreement. The second stage used the best surrogates as well as the target models to generate adversarial malware using the MAB framework to test stand-alone models and AVs (offline and online). Results showed that surrogate models could generate adversarial samples that evade the targets but are less successful than the targets themselves. Using surrogates, however, is a necessity for attackers, given that attacks against AVs are extremely time-consuming and easily detected when the AVs are connected to the internet.



## **5. Liuer Mihou: A Practical Framework for Generating and Evaluating Grey-box Adversarial Attacks against NIDS**

cs.CR

16 pages, 8 figures, planning on submitting to ACM CCS 2022

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2204.06113v1)

**Authors**: Ke He, Dan Dongseong Kim, Jing Sun, Jeong Do Yoo, Young Hun Lee, Huy Kang Kim

**Abstracts**: Due to its high expressiveness and speed, Deep Learning (DL) has become an increasingly popular choice as the detection algorithm for Network-based Intrusion Detection Systems (NIDSes). Unfortunately, DL algorithms are vulnerable to adversarial examples that inject imperceptible modifications to the input and cause the DL algorithm to misclassify the input. Existing adversarial attacks in the NIDS domain often manipulate the traffic features directly, which hold no practical significance because traffic features cannot be replayed in a real network. It remains a research challenge to generate practical and evasive adversarial attacks.   This paper presents the Liuer Mihou attack that generates practical and replayable adversarial network packets that can bypass anomaly-based NIDS deployed in the Internet of Things (IoT) networks. The core idea behind Liuer Mihou is to exploit adversarial transferability and generate adversarial packets on a surrogate NIDS constrained by predefined mutation operations to ensure practicality. We objectively analyse the evasiveness of Liuer Mihou against four ML-based algorithms (LOF, OCSVM, RRCF, and SOM) and the state-of-the-art NIDS, Kitsune. From the results of our experiment, we gain valuable insights into necessary conditions on the adversarial transferability of anomaly detection algorithms. Going beyond a theoretical setting, we replay the adversarial attack in a real IoT testbed to examine the practicality of Liuer Mihou. Furthermore, we demonstrate that existing feature-level adversarial defence cannot defend against Liuer Mihou and constructively criticise the limitations of feature-level adversarial defences.



## **6. Optimal Membership Inference Bounds for Adaptive Composition of Sampled Gaussian Mechanisms**

cs.CR

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2204.06106v1)

**Authors**: Saeed Mahloujifar, Alexandre Sablayrolles, Graham Cormode, Somesh Jha

**Abstracts**: Given a trained model and a data sample, membership-inference (MI) attacks predict whether the sample was in the model's training set. A common countermeasure against MI attacks is to utilize differential privacy (DP) during model training to mask the presence of individual examples. While this use of DP is a principled approach to limit the efficacy of MI attacks, there is a gap between the bounds provided by DP and the empirical performance of MI attacks. In this paper, we derive bounds for the \textit{advantage} of an adversary mounting a MI attack, and demonstrate tightness for the widely-used Gaussian mechanism. We further show bounds on the \textit{confidence} of MI attacks. Our bounds are much stronger than those obtained by DP analysis. For example, analyzing a setting of DP-SGD with $\epsilon=4$ would obtain an upper bound on the advantage of $\approx0.36$ based on our analyses, while getting bound of $\approx 0.97$ using the analysis of previous work that convert $\epsilon$ to membership inference bounds.   Finally, using our analysis, we provide MI metrics for models trained on CIFAR10 dataset. To the best of our knowledge, our analysis provides the state-of-the-art membership inference bounds for the privacy.



## **7. Membership Inference Attacks From First Principles**

cs.CR

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2112.03570v2)

**Authors**: Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramer

**Abstracts**: A membership inference attack allows an adversary to query a trained machine learning model to predict whether or not a particular example was contained in the model's training dataset. These attacks are currently evaluated using average-case "accuracy" metrics that fail to characterize whether the attack can confidently identify any members of the training set. We argue that attacks should instead be evaluated by computing their true-positive rate at low (e.g., <0.1%) false-positive rates, and find most prior attacks perform poorly when evaluated in this way. To address this we develop a Likelihood Ratio Attack (LiRA) that carefully combines multiple ideas from the literature. Our attack is 10x more powerful at low false-positive rates, and also strictly dominates prior attacks on existing metrics.



## **8. Rate Coding or Direct Coding: Which One is Better for Accurate, Robust, and Energy-efficient Spiking Neural Networks?**

cs.NE

Accepted to ICASSP2022

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2202.03133v2)

**Authors**: Youngeun Kim, Hyoungseob Park, Abhishek Moitra, Abhiroop Bhattacharjee, Yeshwanth Venkatesha, Priyadarshini Panda

**Abstracts**: Recent Spiking Neural Networks (SNNs) works focus on an image classification task, therefore various coding techniques have been proposed to convert an image into temporal binary spikes. Among them, rate coding and direct coding are regarded as prospective candidates for building a practical SNN system as they show state-of-the-art performance on large-scale datasets. Despite their usage, there is little attention to comparing these two coding schemes in a fair manner. In this paper, we conduct a comprehensive analysis of the two codings from three perspectives: accuracy, adversarial robustness, and energy-efficiency. First, we compare the performance of two coding techniques with various architectures and datasets. Then, we measure the robustness of the coding techniques on two adversarial attack methods. Finally, we compare the energy-efficiency of two coding schemes on a digital hardware platform. Our results show that direct coding can achieve better accuracy especially for a small number of timesteps. In contrast, rate coding shows better robustness to adversarial attacks owing to the non-differentiable spike generation process. Rate coding also yields higher energy-efficiency than direct coding which requires multi-bit precision for the first layer. Our study explores the characteristics of two codings, which is an important design consideration for building SNNs. The code is made available at https://github.com/Intelligent-Computing-Lab-Yale/Rate-vs-Direct.



## **9. Masked Faces with Faced Masks**

cs.CV

8 pages

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2201.06427v2)

**Authors**: Jiayi Zhu, Qing Guo, Felix Juefei-Xu, Yihao Huang, Yang Liu, Geguang Pu

**Abstracts**: Modern face recognition systems (FRS) still fall short when the subjects are wearing facial masks, a common theme in the age of respiratory pandemics. An intuitive partial remedy is to add a mask detector to flag any masked faces so that the FRS can act accordingly for those low-confidence masked faces. In this work, we set out to investigate the potential vulnerability of such FRS equipped with a mask detector, on large-scale masked faces, which might trigger a serious risk, e.g., letting a suspect evade the FRS where both facial identity and mask are undetected. As existing face recognizers and mask detectors have high performance in their respective tasks, it is significantly challenging to simultaneously fool them and preserve the transferability of the attack. We formulate the new task as the generation of realistic & adversarial-faced mask and make three main contributions: First, we study the naive Delanunay-based masking method (DM) to simulate the process of wearing a faced mask that is cropped from a template image, which reveals the main challenges of this new task. Second, we further equip the DM with the adversarial noise attack and propose the adversarial noise Delaunay-based masking method (AdvNoise-DM) that can fool the face recognition and mask detection effectively but make the face less natural. Third, we propose the adversarial filtering Delaunay-based masking method denoted as MF2M by employing the adversarial filtering for AdvNoise-DM and obtain more natural faces. With the above efforts, the final version not only leads to significant performance deterioration of the state-of-the-art (SOTA) deep learning-based FRS, but also remains undetected by the SOTA facial mask detector, thus successfully fooling both systems at the same time.



## **10. Automated Attacker Synthesis for Distributed Protocols**

cs.CR

24 pages, 15 figures

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2004.01220v4)

**Authors**: Max von Hippel, Cole Vick, Stavros Tripakis, Cristina Nita-Rotaru

**Abstracts**: Distributed protocols should be robust to both benign malfunction (e.g. packet loss or delay) and attacks (e.g. message replay) from internal or external adversaries. In this paper we take a formal approach to the automated synthesis of attackers, i.e. adversarial processes that can cause the protocol to malfunction. Specifically, given a formal threat model capturing the distributed protocol model and network topology, as well as the placement, goals, and interface (inputs and outputs) of potential attackers, we automatically synthesize an attacker. We formalize four attacker synthesis problems - across attackers that always succeed versus those that sometimes fail, and attackers that attack forever versus those that do not - and we propose algorithmic solutions to two of them. We report on a prototype implementation called KORG and its application to TCP as a case-study. Our experiments show that KORG can automatically generate well-known attacks for TCP within seconds or minutes.



## **11. Catch Me If You Can: Blackbox Adversarial Attacks on Automatic Speech Recognition using Frequency Masking**

cs.SD

11 pages, 7 figures and 3 tables

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2112.01821v2)

**Authors**: Xiaoliang Wu, Ajitha Rajan

**Abstracts**: Automatic speech recognition (ASR) models are prevalent, particularly in applications for voice navigation and voice control of domestic appliances. The computational core of ASRs are deep neural networks (DNNs) that have been shown to be susceptible to adversarial perturbations; easily misused by attackers to generate malicious outputs. To help test the security and robustnesss of ASRS, we propose techniques that generate blackbox (agnostic to the DNN), untargeted adversarial attacks that are portable across ASRs. This is in contrast to existing work that focuses on whitebox targeted attacks that are time consuming and lack portability.   Our techniques generate adversarial attacks that have no human audible difference by manipulating the audio signal using a psychoacoustic model that maintains the audio perturbations below the thresholds of human perception. We evaluate portability and effectiveness of our techniques using three popular ASRs and two input audio datasets using the metrics - Word Error Rate (WER) of output transcription, Similarity to original audio, attack Success Rate on different ASRs and Detection score by a defense system. We found our adversarial attacks were portable across ASRs, not easily detected by a state-of-the-art defense system, and had significant difference in output transcriptions while sounding similar to original audio.



## **12. Staircase Sign Method for Boosting Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2104.09722v2)

**Authors**: Qilong Zhang, Xiaosu Zhu, Jingkuan Song, Lianli Gao, Heng Tao Shen

**Abstracts**: Crafting adversarial examples for the transfer-based attack is challenging and remains a research hot spot. Currently, such attack methods are based on the hypothesis that the substitute model and the victim model learn similar decision boundaries, and they conventionally apply Sign Method (SM) to manipulate the gradient as the resultant perturbation. Although SM is efficient, it only extracts the sign of gradient units but ignores their value difference, which inevitably leads to a deviation. Therefore, we propose a novel Staircase Sign Method (S$^2$M) to alleviate this issue, thus boosting attacks. Technically, our method heuristically divides the gradient sign into several segments according to the values of the gradient units, and then assigns each segment with a staircase weight for better crafting adversarial perturbation. As a result, our adversarial examples perform better in both white-box and black-box manner without being more visible. Since S$^2$M just manipulates the resultant gradient, our method can be generally integrated into the family of FGSM algorithms, and the computational overhead is negligible. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our proposed methods, which significantly improve the transferability (i.e., on average, \textbf{5.1\%} for normally trained models and \textbf{12.8\%} for adversarially trained defenses). Our code is available at \url{https://github.com/qilong-zhang/Staircase-sign-method}.



## **13. A survey in Adversarial Defences and Robustness in NLP**

cs.CL

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2203.06414v2)

**Authors**: Shreya Goyal, Sumanth Doddapaneni, Mitesh M. Khapra, Balaraman Ravindran

**Abstracts**: In recent years, it has been seen that deep neural networks are lacking robustness and are likely to break in case of adversarial perturbations in input data. Strong adversarial attacks are proposed by various authors for computer vision and Natural Language Processing (NLP). As a counter-effort, several defense mechanisms are also proposed to save these networks from failing. In contrast with image data, generating adversarial attacks and defending these models is not easy in NLP because of the discrete nature of the text data. However, numerous methods for adversarial defense are proposed of late, for different NLP tasks such as text classification, named entity recognition, natural language inferencing, etc. These methods are not just used for defending neural networks from adversarial attacks, but also used as a regularization mechanism during training, saving the model from overfitting. The proposed survey is an attempt to review different methods proposed for adversarial defenses in NLP in the recent past by proposing a novel taxonomy. This survey also highlights the fragility of the advanced deep neural networks in NLP and the challenges in defending them.



## **14. Generalizing Adversarial Explanations with Grad-CAM**

cs.CV

Accepted in CVPRw ArtofRobustness workshop

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.05427v1)

**Authors**: Tanmay Chakraborty, Utkarsh Trehan, Khawla Mallat, Jean-Luc Dugelay

**Abstracts**: Gradient-weighted Class Activation Mapping (Grad- CAM), is an example-based explanation method that provides a gradient activation heat map as an explanation for Convolution Neural Network (CNN) models. The drawback of this method is that it cannot be used to generalize CNN behaviour. In this paper, we present a novel method that extends Grad-CAM from example-based explanations to a method for explaining global model behaviour. This is achieved by introducing two new metrics, (i) Mean Observed Dissimilarity (MOD) and (ii) Variation in Dissimilarity (VID), for model generalization. These metrics are computed by comparing a Normalized Inverted Structural Similarity Index (NISSIM) metric of the Grad-CAM generated heatmap for samples from the original test set and samples from the adversarial test set. For our experiment, we study adversarial attacks on deep models such as VGG16, ResNet50, and ResNet101, and wide models such as InceptionNetv3 and XceptionNet using Fast Gradient Sign Method (FGSM). We then compute the metrics MOD and VID for the automatic face recognition (AFR) use case with the VGGFace2 dataset. We observe a consistent shift in the region highlighted in the Grad-CAM heatmap, reflecting its participation to the decision making, across all models under adversarial attacks. The proposed method can be used to understand adversarial attacks and explain the behaviour of black box CNN models for image analysis.



## **15. Segmentation-Consistent Probabilistic Lesion Counting**

eess.IV

Accepted at Medical Imaging with Deep Learning (MIDL) 2022

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.05276v1)

**Authors**: Julien Schroeter, Chelsea Myers-Colet, Douglas L Arnold, Tal Arbel

**Abstracts**: Lesion counts are important indicators of disease severity, patient prognosis, and treatment efficacy, yet counting as a task in medical imaging is often overlooked in favor of segmentation. This work introduces a novel continuously differentiable function that maps lesion segmentation predictions to lesion count probability distributions in a consistent manner. The proposed end-to-end approach--which consists of voxel clustering, lesion-level voxel probability aggregation, and Poisson-binomial counting--is non-parametric and thus offers a robust and consistent way to augment lesion segmentation models with post hoc counting capabilities. Experiments on Gadolinium-enhancing lesion counting demonstrate that our method outputs accurate and well-calibrated count distributions that capture meaningful uncertainty information. They also reveal that our model is suitable for multi-task learning of lesion segmentation, is efficient in low data regimes, and is robust to adversarial attacks.



## **16. Exploring the Universal Vulnerability of Prompt-based Learning Paradigm**

cs.CL

Accepted to Findings of NAACL 2022

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.05239v1)

**Authors**: Lei Xu, Yangyi Chen, Ganqu Cui, Hongcheng Gao, Zhiyuan Liu

**Abstracts**: Prompt-based learning paradigm bridges the gap between pre-training and fine-tuning, and works effectively under the few-shot setting. However, we find that this learning paradigm inherits the vulnerability from the pre-training stage, where model predictions can be misled by inserting certain triggers into the text. In this paper, we explore this universal vulnerability by either injecting backdoor triggers or searching for adversarial triggers on pre-trained language models using only plain text. In both scenarios, we demonstrate that our triggers can totally control or severely decrease the performance of prompt-based models fine-tuned on arbitrary downstream tasks, reflecting the universal vulnerability of the prompt-based learning paradigm. Further experiments show that adversarial triggers have good transferability among language models. We also find conventional fine-tuning models are not vulnerable to adversarial triggers constructed from pre-trained language models. We conclude by proposing a potential solution to mitigate our attack methods. Code and data are publicly available at https://github.com/leix28/prompt-universal-vulnerability



## **17. Analysis of a blockchain protocol based on LDPC codes**

cs.CR

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2202.07265v2)

**Authors**: Massimo Battaglioni, Paolo Santini, Giulia Rafaiani, Franco Chiaraluce, Marco Baldi

**Abstracts**: In a blockchain Data Availability Attack (DAA), a malicious node publishes a block header but withholds part of the block, which contains invalid transactions. Honest full nodes, which can download and store the full blockchain, are aware that some data are not available but they have no formal way to prove it to light nodes, i.e., nodes that have limited resources and are not able to access the whole blockchain data. A common solution to counter these attacks exploits linear error correcting codes to encode the block content. A recent protocol, called SPAR, employs coded Merkle trees and low-density parity-check (LDPC) codes to counter DAAs. We show that the protocol is less secure than expected, owing to a redefinition of the adversarial success probability.



## **18. Measuring and Mitigating the Risk of IP Reuse on Public Clouds**

cs.CR

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.05122v1)

**Authors**: Eric Pauley, Ryan Sheatsley, Blaine Hoak, Quinn Burke, Yohan Beugin, Patrick McDaniel

**Abstracts**: Public clouds provide scalable and cost-efficient computing through resource sharing. However, moving from traditional on-premises service management to clouds introduces new challenges; failure to correctly provision, maintain, or decommission elastic services can lead to functional failure and vulnerability to attack. In this paper, we explore a broad class of attacks on clouds which we refer to as cloud squatting. In a cloud squatting attack, an adversary allocates resources in the cloud (e.g., IP addresses) and thereafter leverages latent configuration to exploit prior tenants. To measure and categorize cloud squatting we deployed a custom Internet telescope within the Amazon Web Services us-east-1 region. Using this apparatus, we deployed over 3 million servers receiving 1.5 million unique IP addresses (56% of the available pool) over 101 days beginning in March of 2021. We identified 4 classes of cloud services, 7 classes of third-party services, and DNS as sources of exploitable latent configurations. We discovered that exploitable configurations were both common and in many cases extremely dangerous; we received over 5 million cloud messages, many containing sensitive data such as financial transactions, GPS location, and PII. Within the 7 classes of third-party services, we identified dozens of exploitable software systems spanning hundreds of servers (e.g., databases, caches, mobile applications, and web services). Lastly, we identified 5446 exploitable domains spanning 231 eTLDs-including 105 in the top 10,000 and 23 in the top 1000 popular domains. Through tenant disclosures we have identified several root causes, including (a) a lack of organizational controls, (b) poor service hygiene, and (c) failure to follow best practices. We conclude with a discussion of the space of possible mitigations and describe the mitigations to be deployed by Amazon in response to this study.



## **19. Anti-Adversarially Manipulated Attributions for Weakly Supervised Semantic Segmentation and Object Localization**

cs.CV

IEEE TPAMI, 2022

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.04890v1)

**Authors**: Jungbeom Lee, Eunji Kim, Jisoo Mok, Sungroh Yoon

**Abstracts**: Obtaining accurate pixel-level localization from class labels is a crucial process in weakly supervised semantic segmentation and object localization. Attribution maps from a trained classifier are widely used to provide pixel-level localization, but their focus tends to be restricted to a small discriminative region of the target object. An AdvCAM is an attribution map of an image that is manipulated to increase the classification score produced by a classifier before the final softmax or sigmoid layer. This manipulation is realized in an anti-adversarial manner, so that the original image is perturbed along pixel gradients in directions opposite to those used in an adversarial attack. This process enhances non-discriminative yet class-relevant features, which make an insufficient contribution to previous attribution maps, so that the resulting AdvCAM identifies more regions of the target object. In addition, we introduce a new regularization procedure that inhibits the incorrect attribution of regions unrelated to the target object and the excessive concentration of attributions on a small region of the target object. Our method achieves a new state-of-the-art performance in weakly and semi-supervised semantic segmentation, on both the PASCAL VOC 2012 and MS COCO 2014 datasets. In weakly supervised object localization, it achieves a new state-of-the-art performance on the CUB-200-2011 and ImageNet-1K datasets.



## **20. Adversarial Robustness of Deep Sensor Fusion Models**

cs.CV

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2006.13192v3)

**Authors**: Shaojie Wang, Tong Wu, Ayan Chakrabarti, Yevgeniy Vorobeychik

**Abstracts**: We experimentally study the robustness of deep camera-LiDAR fusion architectures for 2D object detection in autonomous driving. First, we find that the fusion model is usually both more accurate, and more robust against single-source attacks than single-sensor deep neural networks. Furthermore, we show that without adversarial training, early fusion is more robust than late fusion, whereas the two perform similarly after adversarial training. However, we note that single-channel adversarial training of deep fusion is often detrimental even to robustness. Moreover, we observe cross-channel externalities, where single-channel adversarial training reduces robustness to attacks on the other channel. Additionally, we observe that the choice of adversarial model in adversarial training is critical: using attacks restricted to cars' bounding boxes is more effective in adversarial training and exhibits less significant cross-channel externalities. Finally, we find that joint-channel adversarial training helps mitigate many of the issues above, but does not significantly boost adversarial robustness.



## **21. Measuring the False Sense of Security**

cs.LG

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2204.04778v1)

**Authors**: Carlos Gomes

**Abstracts**: Recently, several papers have demonstrated how widespread gradient masking is amongst proposed adversarial defenses. Defenses that rely on this phenomenon are considered failed, and can easily be broken. Despite this, there has been little investigation into ways of measuring the phenomenon of gradient masking and enabling comparisons of its extent amongst different networks. In this work, we investigate gradient masking under the lens of its mensurability, departing from the idea that it is a binary phenomenon. We propose and motivate several metrics for it, performing extensive empirical tests on defenses suspected of exhibiting different degrees of gradient masking. These are computationally cheaper than strong attacks, enable comparisons between models, and do not require the large time investment of tailor-made attacks for specific models. Our results reveal metrics that are successful in measuring the extent of gradient masking across different networks



## **22. Analysis of Power-Oriented Fault Injection Attacks on Spiking Neural Networks**

cs.AI

Design, Automation and Test in Europe Conference (DATE) 2022

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2204.04768v1)

**Authors**: Karthikeyan Nagarajan, Junde Li, Sina Sayyah Ensan, Mohammad Nasim Imtiaz Khan, Sachhidh Kannan, Swaroop Ghosh

**Abstracts**: Spiking Neural Networks (SNN) are quickly gaining traction as a viable alternative to Deep Neural Networks (DNN). In comparison to DNNs, SNNs are more computationally powerful and provide superior energy efficiency. SNNs, while exciting at first appearance, contain security-sensitive assets (e.g., neuron threshold voltage) and vulnerabilities (e.g., sensitivity of classification accuracy to neuron threshold voltage change) that adversaries can exploit. We investigate global fault injection attacks by employing external power supplies and laser-induced local power glitches to corrupt crucial training parameters such as spike amplitude and neuron's membrane threshold potential on SNNs developed using common analog neurons. We also evaluate the impact of power-based attacks on individual SNN layers for 0% (i.e., no attack) to 100% (i.e., whole layer under attack). We investigate the impact of the attacks on digit classification tasks and find that in the worst-case scenario, classification accuracy is reduced by 85.65%. We also propose defenses e.g., a robust current driver design that is immune to power-oriented attacks, improved circuit sizing of neuron components to reduce/recover the adversarial accuracy degradation at the cost of negligible area and 25% power overhead. We also present a dummy neuron-based voltage fault injection detection system with 1% power and area overhead.



## **23. "That Is a Suspicious Reaction!": Interpreting Logits Variation to Detect NLP Adversarial Attacks**

cs.AI

ACL 2022

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2204.04636v1)

**Authors**: Edoardo Mosca, Shreyash Agarwal, Javier Rando-Ramirez, Georg Groh

**Abstracts**: Adversarial attacks are a major challenge faced by current machine learning research. These purposely crafted inputs fool even the most advanced models, precluding their deployment in safety-critical applications. Extensive research in computer vision has been carried to develop reliable defense strategies. However, the same issue remains less explored in natural language processing. Our work presents a model-agnostic detector of adversarial text examples. The approach identifies patterns in the logits of the target classifier when perturbing the input text. The proposed detector improves the current state-of-the-art performance in recognizing adversarial inputs and exhibits strong generalization capabilities across different NLP models, datasets, and word-level attacks.



## **24. LTD: Low Temperature Distillation for Robust Adversarial Training**

cs.CV

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2111.02331v2)

**Authors**: Erh-Chung Chen, Che-Rung Lee

**Abstracts**: Adversarial training has been widely used to enhance the robustness of the neural network models against adversarial attacks. However, there still a notable gap between the nature accuracy and the robust accuracy. We found one of the reasons is the commonly used labels, one-hot vectors, hinder the learning process for image recognition. In this paper, we proposed a method, called Low Temperature Distillation (LTD), which is based on the knowledge distillation framework to generate the desired soft labels. Unlike the previous work, LTD uses relatively low temperature in the teacher model, and employs different, but fixed, temperatures for the teacher model and the student model. Moreover, we have investigated the methods to synergize the use of nature data and adversarial ones in LTD. Experimental results show that without extra unlabeled data, the proposed method combined with the previous work can achieve 57.72\% and 30.36\% robust accuracy on CIFAR-10 and CIFAR-100 dataset respectively, which is about 1.21\% improvement of the state-of-the-art methods in average.



## **25. Understanding, Detecting, and Separating Out-of-Distribution Samples and Adversarial Samples in Text Classification**

cs.CL

Preprint. Work in progress

**SubmitDate**: 2022-04-09    [paper-pdf](http://arxiv.org/pdf/2204.04458v1)

**Authors**: Cheng-Han Chiang, Hung-yi Lee

**Abstracts**: In this paper, we study the differences and commonalities between statistically out-of-distribution (OOD) samples and adversarial (Adv) samples, both of which hurting a text classification model's performance. We conduct analyses to compare the two types of anomalies (OOD and Adv samples) with the in-distribution (ID) ones from three aspects: the input features, the hidden representations in each layer of the model, and the output probability distributions of the classifier. We find that OOD samples expose their aberration starting from the first layer, while the abnormalities of Adv samples do not emerge until the deeper layers of the model. We also illustrate that the models' output probabilities for Adv samples tend to be more unconfident. Based on our observations, we propose a simple method to separate ID, OOD, and Adv samples using the hidden representations and output probabilities of the model. On multiple combinations of ID, OOD datasets, and Adv attacks, our proposed method shows exceptional results on distinguishing ID, OOD, and Adv samples.



## **26. PatchCleanser: Certifiably Robust Defense against Adversarial Patches for Any Image Classifier**

cs.CV

USENIX Security Symposium 2022; extended technical report

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2108.09135v2)

**Authors**: Chong Xiang, Saeed Mahloujifar, Prateek Mittal

**Abstracts**: The adversarial patch attack against image classification models aims to inject adversarially crafted pixels within a restricted image region (i.e., a patch) for inducing model misclassification. This attack can be realized in the physical world by printing and attaching the patch to the victim object; thus, it imposes a real-world threat to computer vision systems. To counter this threat, we design PatchCleanser as a certifiably robust defense against adversarial patches. In PatchCleanser, we perform two rounds of pixel masking on the input image to neutralize the effect of the adversarial patch. This image-space operation makes PatchCleanser compatible with any state-of-the-art image classifier for achieving high accuracy. Furthermore, we can prove that PatchCleanser will always predict the correct class labels on certain images against any adaptive white-box attacker within our threat model, achieving certified robustness. We extensively evaluate PatchCleanser on the ImageNet, ImageNette, CIFAR-10, CIFAR-100, SVHN, and Flowers-102 datasets and demonstrate that our defense achieves similar clean accuracy as state-of-the-art classification models and also significantly improves certified robustness from prior works. Remarkably, PatchCleanser achieves 83.9% top-1 clean accuracy and 62.1% top-1 certified robust accuracy against a 2%-pixel square patch anywhere on the image for the 1000-class ImageNet dataset.



## **27. Path Defense in Dynamic Defender-Attacker Blotto Games (dDAB) with Limited Information**

cs.GT

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.04176v1)

**Authors**: Austin K. Chen, Bryce L. Ferguson, Daigo Shishika, Michael Dorothy, Jason R. Marden, George J. Pappas, Vijay Kumar

**Abstracts**: We consider a path guarding problem in dynamic Defender-Attacker Blotto games (dDAB), where a team of robots must defend a path in a graph against adversarial agents. Multi-robot systems are particularly well suited to this application, as recent work has shown the effectiveness of these systems in related areas such as perimeter defense and surveillance. When designing a defender policy that guarantees the defense of a path, information about the adversary and the environment can be helpful and may reduce the number of resources required by the defender to achieve a sufficient level of security. In this work, we characterize the necessary and sufficient number of assets needed to guarantee the defense of a shortest path between two nodes in dDAB games when the defender can only detect assets within $k$-hops of a shortest path. By characterizing the relationship between sensing horizon and required resources, we show that increasing the sensing capability of the defender greatly reduces the number of defender assets needed to defend the path.



## **28. DAD: Data-free Adversarial Defense at Test Time**

cs.LG

WACV 2022. Project page: https://sites.google.com/view/dad-wacv22

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.01568v2)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Anirban Chakraborty

**Abstracts**: Deep models are highly susceptible to adversarial attacks. Such attacks are carefully crafted imperceptible noises that can fool the network and can cause severe consequences when deployed. To encounter them, the model requires training data for adversarial training or explicit regularization-based techniques. However, privacy has become an important concern, restricting access to only trained models but not the training data (e.g. biometric data). Also, data curation is expensive and companies may have proprietary rights over it. To handle such situations, we propose a completely novel problem of 'test-time adversarial defense in absence of training data and even their statistics'. We solve it in two stages: a) detection and b) correction of adversarial samples. Our adversarial sample detection framework is initially trained on arbitrary data and is subsequently adapted to the unlabelled test data through unsupervised domain adaptation. We further correct the predictions on detected adversarial samples by transforming them in Fourier domain and obtaining their low frequency component at our proposed suitable radius for model prediction. We demonstrate the efficacy of our proposed technique via extensive experiments against several adversarial attacks and for different model architectures and datasets. For a non-robust Resnet-18 model pre-trained on CIFAR-10, our detection method correctly identifies 91.42% adversaries. Also, we significantly improve the adversarial accuracy from 0% to 37.37% with a minimal drop of 0.02% in clean accuracy on state-of-the-art 'Auto Attack' without having to retrain the model.



## **29. Training strategy for a lightweight countermeasure model for automatic speaker verification**

cs.SD

ASVspoof2021

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2203.17031v3)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstracts**: The countermeasure (CM) model is developed to protect Automatic Speaker Verification (ASV) systems from spoof attacks and prevent resulting personal information leakage. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud-based systems. This work proposes training strategies for a lightweight CM model for ASV, using generalized end-to-end (GE2E) pre-training and adversarial fine-tuning to improve performance, and applying knowledge distillation (KD) to reduce the size of the CM model. In the evaluation phase of the ASVspoof 2021 Logical Access task, the lightweight ResNetSE model reaches min t-DCF 0.2695 and EER 3.54%. Compared to the teacher model, the lightweight student model only uses 22.5% of parameters and 21.1% of multiply and accumulate operands of the teacher model.



## **30. Spinning Language Models: Risks of Propaganda-As-A-Service and Countermeasures**

cs.CR

IEEE S&P 2022. arXiv admin note: text overlap with arXiv:2107.10443

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2112.05224v2)

**Authors**: Eugene Bagdasaryan, Vitaly Shmatikov

**Abstracts**: We investigate a new threat to neural sequence-to-sequence (seq2seq) models: training-time attacks that cause models to "spin" their outputs so as to support an adversary-chosen sentiment or point of view -- but only when the input contains adversary-chosen trigger words. For example, a spinned summarization model outputs positive summaries of any text that mentions the name of some individual or organization.   Model spinning introduces a "meta-backdoor" into a model. Whereas conventional backdoors cause models to produce incorrect outputs on inputs with the trigger, outputs of spinned models preserve context and maintain standard accuracy metrics, yet also satisfy a meta-task chosen by the adversary.   Model spinning enables propaganda-as-a-service, where propaganda is defined as biased speech. An adversary can create customized language models that produce desired spins for chosen triggers, then deploy these models to generate disinformation (a platform attack), or else inject them into ML training pipelines (a supply-chain attack), transferring malicious functionality to downstream models trained by victims.   To demonstrate the feasibility of model spinning, we develop a new backdooring technique. It stacks an adversarial meta-task onto a seq2seq model, backpropagates the desired meta-task output to points in the word-embedding space we call "pseudo-words," and uses pseudo-words to shift the entire output distribution of the seq2seq model. We evaluate this attack on language generation, summarization, and translation models with different triggers and meta-tasks such as sentiment, toxicity, and entailment. Spinned models largely maintain their accuracy metrics (ROUGE and BLEU) while shifting their outputs to satisfy the adversary's meta-task. We also show that, in the case of a supply-chain attack, the spin functionality transfers to downstream models.



## **31. Backdoor Attack against NLP models with Robustness-Aware Perturbation defense**

cs.CR

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.05758v1)

**Authors**: Shaik Mohammed Maqsood, Viveros Manuela Ceron, Addluri GowthamKrishna

**Abstracts**: Backdoor attack intends to embed hidden backdoor into deep neural networks (DNNs), such that the attacked model performs well on benign samples, whereas its prediction will be maliciously changed if the hidden backdoor is activated by the attacker defined trigger. This threat could happen when the training process is not fully controlled, such as training on third-party data-sets or adopting third-party models. There has been a lot of research and different methods to defend such type of backdoor attacks, one being robustness-aware perturbation-based defense method. This method mainly exploits big gap of robustness between poisoned and clean samples. In our work, we break this defense by controlling the robustness gap between poisoned and clean samples using adversarial training step.



## **32. Defense against Adversarial Attacks on Hybrid Speech Recognition using Joint Adversarial Fine-tuning with Denoiser**

eess.AS

Submitted to Interspeech 2022

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.03851v1)

**Authors**: Sonal Joshi, Saurabh Kataria, Yiwen Shao, Piotr Zelasko, Jesus Villalba, Sanjeev Khudanpur, Najim Dehak

**Abstracts**: Adversarial attacks are a threat to automatic speech recognition (ASR) systems, and it becomes imperative to propose defenses to protect them. In this paper, we perform experiments to show that K2 conformer hybrid ASR is strongly affected by white-box adversarial attacks. We propose three defenses--denoiser pre-processor, adversarially fine-tuning ASR model, and adversarially fine-tuning joint model of ASR and denoiser. Our evaluation shows denoiser pre-processor (trained on offline adversarial examples) fails to defend against adaptive white-box attacks. However, adversarially fine-tuning the denoiser using a tandem model of denoiser and ASR offers more robustness. We evaluate two variants of this defense--one updating parameters of both models and the second keeping ASR frozen. The joint model offers a mean absolute decrease of 19.3\% ground truth (GT) WER with reference to baseline against fast gradient sign method (FGSM) attacks with different $L_\infty$ norms. The joint model with frozen ASR parameters gives the best defense against projected gradient descent (PGD) with 7 iterations, yielding a mean absolute increase of 22.3\% GT WER with reference to baseline; and against PGD with 500 iterations, yielding a mean absolute decrease of 45.08\% GT WER and an increase of 68.05\% adversarial target WER.



## **33. AdvEst: Adversarial Perturbation Estimation to Classify and Detect Adversarial Attacks against Speaker Identification**

eess.AS

Submitted to InterSpeech 2022

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.03848v1)

**Authors**: Sonal Joshi, Saurabh Kataria, Jesus Villalba, Najim Dehak

**Abstracts**: Adversarial attacks pose a severe security threat to the state-of-the-art speaker identification systems, thereby making it vital to propose countermeasures against them. Building on our previous work that used representation learning to classify and detect adversarial attacks, we propose an improvement to it using AdvEst, a method to estimate adversarial perturbation. First, we prove our claim that training the representation learning network using adversarial perturbations as opposed to adversarial examples (consisting of the combination of clean signal and adversarial perturbation) is beneficial because it eliminates nuisance information. At inference time, we use a time-domain denoiser to estimate the adversarial perturbations from adversarial examples. Using our improved representation learning approach to obtain attack embeddings (signatures), we evaluate their performance for three applications: known attack classification, attack verification, and unknown attack detection. We show that common attacks in the literature (Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), Carlini-Wagner (CW) with different Lp threat models) can be classified with an accuracy of ~96%. We also detect unknown attacks with an equal error rate (EER) of ~9%, which is absolute improvement of ~12% from our previous work.



## **34. Using Multiple Self-Supervised Tasks Improves Model Robustness**

cs.CV

Accepted to ICLR 2022 Workshop on PAIR^2Struct: Privacy,  Accountability, Interpretability, Robustness, Reasoning on Structured Data

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2204.03714v1)

**Authors**: Matthew Lawhon, Chengzhi Mao, Junfeng Yang

**Abstracts**: Deep networks achieve state-of-the-art performance on computer vision tasks, yet they fail under adversarial attacks that are imperceptible to humans. In this paper, we propose a novel defense that can dynamically adapt the input using the intrinsic structure from multiple self-supervised tasks. By simultaneously using many self-supervised tasks, our defense avoids over-fitting the adapted image to one specific self-supervised task and restores more intrinsic structure in the image compared to a single self-supervised task approach. Our approach further improves robustness and clean accuracy significantly compared to the state-of-the-art single task self-supervised defense. Our work is the first to connect multiple self-supervised tasks to robustness, and suggests that we can achieve better robustness with more intrinsic signal from visual data.



## **35. Adaptive-Gravity: A Defense Against Adversarial Samples**

cs.LG

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2204.03694v1)

**Authors**: Ali Mirzaeian, Zhi Tian, Sai Manoj P D, Banafsheh S. Latibari, Ioannis Savidis, Houman Homayoun, Avesta Sasan

**Abstracts**: This paper presents a novel model training solution, denoted as Adaptive-Gravity, for enhancing the robustness of deep neural network classifiers against adversarial examples. We conceptualize the model parameters/features associated with each class as a mass characterized by its centroid location and the spread (standard deviation of the distance) of features around the centroid. We use the centroid associated with each cluster to derive an anti-gravity force that pushes the centroids of different classes away from one another during network training. Then we customized an objective function that aims to concentrate each class's features toward their corresponding new centroid, which has been obtained by anti-gravity force. This methodology results in a larger separation between different masses and reduces the spread of features around each centroid. As a result, the samples are pushed away from the space that adversarial examples could be mapped to, effectively increasing the degree of perturbation needed for making an adversarial example. We have implemented this training solution as an iterative method consisting of four steps at each iteration: 1) centroid extraction, 2) anti-gravity force calculation, 3) centroid relocation, and 4) gravity training. Gravity's efficiency is evaluated by measuring the corresponding fooling rates against various attack models, including FGSM, MIM, BIM, and PGD using LeNet and ResNet110 networks, benchmarked against MNIST and CIFAR10 classification problems. Test results show that Gravity not only functions as a powerful instrument to robustify a model against state-of-the-art adversarial attacks but also effectively improves the model training accuracy.



## **36. Reinforcement Learning for Linear Quadratic Control is Vulnerable Under Cost Manipulation**

eess.SY

This paper is yet to be peer-reviewed; Typos are corrected in ver 2

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2203.05774v2)

**Authors**: Yunhan Huang, Quanyan Zhu

**Abstracts**: In this work, we study the deception of a Linear-Quadratic-Gaussian (LQG) agent by manipulating the cost signals. We show that a small falsification of the cost parameters will only lead to a bounded change in the optimal policy. The bound is linear on the amount of falsification the attacker can apply to the cost parameters. We propose an attack model where the attacker aims to mislead the agent into learning a `nefarious' policy by intentionally falsifying the cost parameters. We formulate the attack's problem as a convex optimization problem and develop necessary and sufficient conditions to check the achievability of the attacker's goal.   We showcase the adversarial manipulation on two types of LQG learners: the batch RL learner and the other is the adaptive dynamic programming (ADP) learner. Our results demonstrate that with only 2.296% of falsification on the cost data, the attacker misleads the batch RL into learning the 'nefarious' policy that leads the vehicle to a dangerous position. The attacker can also gradually trick the ADP learner into learning the same `nefarious' policy by consistently feeding the learner a falsified cost signal that stays close to the actual cost signal. The paper aims to raise people's awareness of the security threats faced by RL-enabled control systems.



## **37. IRShield: A Countermeasure Against Adversarial Physical-Layer Wireless Sensing**

cs.CR

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2112.01967v2)

**Authors**: Paul Staat, Simon Mulzer, Stefan Roth, Veelasha Moonsamy, Markus Heinrichs, Rainer Kronberger, Aydin Sezgin, Christof Paar

**Abstracts**: Wireless radio channels are known to contain information about the surrounding propagation environment, which can be extracted using established wireless sensing methods. Thus, today's ubiquitous wireless devices are attractive targets for passive eavesdroppers to launch reconnaissance attacks. In particular, by overhearing standard communication signals, eavesdroppers obtain estimations of wireless channels which can give away sensitive information about indoor environments. For instance, by applying simple statistical methods, adversaries can infer human motion from wireless channel observations, allowing to remotely monitor premises of victims. In this work, building on the advent of intelligent reflecting surfaces (IRSs), we propose IRShield as a novel countermeasure against adversarial wireless sensing. IRShield is designed as a plug-and-play privacy-preserving extension to existing wireless networks. At the core of IRShield, we design an IRS configuration algorithm to obfuscate wireless channels. We validate the effectiveness with extensive experimental evaluations. In a state-of-the-art human motion detection attack using off-the-shelf Wi-Fi devices, IRShield lowered detection rates to 5% or less.



## **38. Transfer Attacks Revisited: A Large-Scale Empirical Study in Real Computer Vision Settings**

cs.CV

Accepted to IEEE Security & Privacy 2022

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2204.04063v1)

**Authors**: Yuhao Mao, Chong Fu, Saizhuo Wang, Shouling Ji, Xuhong Zhang, Zhenguang Liu, Jun Zhou, Alex X. Liu, Raheem Beyah, Ting Wang

**Abstracts**: One intriguing property of adversarial attacks is their "transferability" -- an adversarial example crafted with respect to one deep neural network (DNN) model is often found effective against other DNNs as well. Intensive research has been conducted on this phenomenon under simplistic controlled conditions. Yet, thus far, there is still a lack of comprehensive understanding about transferability-based attacks ("transfer attacks") in real-world environments.   To bridge this critical gap, we conduct the first large-scale systematic empirical study of transfer attacks against major cloud-based MLaaS platforms, taking the components of a real transfer attack into account. The study leads to a number of interesting findings which are inconsistent to the existing ones, including: (1) Simple surrogates do not necessarily improve real transfer attacks. (2) No dominant surrogate architecture is found in real transfer attacks. (3) It is the gap between posterior (output of the softmax layer) rather than the gap between logit (so-called $\kappa$ value) that increases transferability. Moreover, by comparing with prior works, we demonstrate that transfer attacks possess many previously unknown properties in real-world environments, such as (1) Model similarity is not a well-defined concept. (2) $L_2$ norm of perturbation can generate high transferability without usage of gradient and is a more powerful source than $L_\infty$ norm. We believe this work sheds light on the vulnerabilities of popular MLaaS platforms and points to a few promising research directions.



## **39. Adversarial Machine Learning Attacks Against Video Anomaly Detection Systems**

cs.CV

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2204.03141v1)

**Authors**: Furkan Mumcu, Keval Doshi, Yasin Yilmaz

**Abstracts**: Anomaly detection in videos is an important computer vision problem with various applications including automated video surveillance. Although adversarial attacks on image understanding models have been heavily investigated, there is not much work on adversarial machine learning targeting video understanding models and no previous work which focuses on video anomaly detection. To this end, we investigate an adversarial machine learning attack against video anomaly detection systems, that can be implemented via an easy-to-perform cyber-attack. Since surveillance cameras are usually connected to the server running the anomaly detection model through a wireless network, they are prone to cyber-attacks targeting the wireless connection. We demonstrate how Wi-Fi deauthentication attack, a notoriously easy-to-perform and effective denial-of-service (DoS) attack, can be utilized to generate adversarial data for video anomaly detection systems. Specifically, we apply several effects caused by the Wi-Fi deauthentication attack on video quality (e.g., slow down, freeze, fast forward, low resolution) to the popular benchmark datasets for video anomaly detection. Our experiments with several state-of-the-art anomaly detection models show that the attackers can significantly undermine the reliability of video anomaly detection systems by causing frequent false alarms and hiding physical anomalies from the surveillance system.



## **40. Control barrier function based attack-recovery with provable guarantees**

cs.SY

8 pages, 6 figures

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.03077v1)

**Authors**: Kunal Garg, Ricardo G. Sanfelice, Alvaro A. Cardenas

**Abstracts**: This paper studies provable security guarantees for cyber-physical systems (CPS) under actuator attacks. In particular, we consider CPS safety and propose a new attack-detection mechanism based on a zeroing control barrier function (ZCBF) condition. In addition we design an adaptive recovery mechanism based on how close the system is from violating safety. We show that the attack-detection mechanism is sound, i.e., there are no false negatives for adversarial attacks. Finally, we use a Quadratic Programming (QP) approach for online recovery (and nominal) control synthesis. We demonstrate the effectiveness of the proposed method in a simulation case study involving a quadrotor with an attack on its motors.



## **41. Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02887v1)

**Authors**: Xu Han, Anmin Liu, Yifeng Xiong, Yanbo Fan, Kun He

**Abstracts**: Deep neural networks have shown to be very vulnerable to adversarial examples crafted by adding human-imperceptible perturbations to benign inputs. After achieving impressive attack success rates in the white-box setting, more focus is shifted to black-box attacks. In either case, the common gradient-based approaches generally use the $sign$ function to generate perturbations at the end of the process. However, only a few works pay attention to the limitation of the $sign$ function. Deviation between the original gradient and the generated noises may lead to inaccurate gradient update estimation and suboptimal solutions for adversarial transferability, which is crucial for black-box attacks. To address this issue, we propose a Sampling-based Fast Gradient Rescaling Method (S-FGRM) to improve the transferability of the crafted adversarial examples. Specifically, we use data rescaling to substitute the inefficient $sign$ function in gradient-based attacks without extra computational cost. We also propose a Depth First Sampling method to eliminate the fluctuation of rescaling and stabilize the gradient update. Our method can be used in any gradient-based optimizations and is extensible to be integrated with various input transformation or ensemble methods for further improving the adversarial transferability. Extensive experiments on the standard ImageNet dataset show that our S-FGRM could significantly boost the transferability of gradient-based attacks and outperform the state-of-the-art baselines.



## **42. Distilling Robust and Non-Robust Features in Adversarial Examples by Information Bottleneck**

cs.LG

NeurIPS 2021

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02735v1)

**Authors**: Junho Kim, Byung-Kwan Lee, Yong Man Ro

**Abstracts**: Adversarial examples, generated by carefully crafted perturbation, have attracted considerable attention in research fields. Recent works have argued that the existence of the robust and non-robust features is a primary cause of the adversarial examples, and investigated their internal interactions in the feature space. In this paper, we propose a way of explicitly distilling feature representation into the robust and non-robust features, using Information Bottleneck. Specifically, we inject noise variation to each feature unit and evaluate the information flow in the feature representation to dichotomize feature units either robust or non-robust, based on the noise variation magnitude. Through comprehensive experiments, we demonstrate that the distilled features are highly correlated with adversarial prediction, and they have human-perceptible semantic information by themselves. Furthermore, we present an attack mechanism intensifying the gradient of non-robust features that is directly related to the model prediction, and validate its effectiveness of breaking model robustness.



## **43. Rolling Colors: Adversarial Laser Exploits against Traffic Light Recognition**

cs.CV

To be published in USENIX Security 2022

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02675v1)

**Authors**: Chen Yan, Zhijian Xu, Zhanyuan Yin, Xiaoyu Ji, Wenyuan Xu

**Abstracts**: Traffic light recognition is essential for fully autonomous driving in urban areas. In this paper, we investigate the feasibility of fooling traffic light recognition mechanisms by shedding laser interference on the camera. By exploiting the rolling shutter of CMOS sensors, we manage to inject a color stripe overlapped on the traffic light in the image, which can cause a red light to be recognized as a green light or vice versa. To increase the success rate, we design an optimization method to search for effective laser parameters based on empirical models of laser interference. Our evaluation in emulated and real-world setups on 2 state-of-the-art recognition systems and 5 cameras reports a maximum success rate of 30% and 86.25% for Red-to-Green and Green-to-Red attacks. We observe that the attack is effective in continuous frames from more than 40 meters away against a moving vehicle, which may cause end-to-end impacts on self-driving such as running a red light or emergency stop. To mitigate the threat, we propose redesigning the rolling shutter mechanism.



## **44. Adversarial Analysis of the Differentially-Private Federated Learning in Cyber-Physical Critical Infrastructures**

cs.CR

11 pages, 5 figures, 4 tables. This work has been submitted to IEEE  for possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02654v1)

**Authors**: Md Tamjid Hossain, Shahriar Badsha, Hung, La, Haoting Shen, Shafkat Islam, Ibrahim Khalil, Xun Yi

**Abstracts**: Differential privacy (DP) is considered to be an effective privacy-preservation method to secure the promising distributed machine learning (ML) paradigm-federated learning (FL) from privacy attacks (e.g., membership inference attack). Nevertheless, while the DP mechanism greatly alleviates privacy concerns, recent studies have shown that it can be exploited to conduct security attacks (e.g., false data injection attacks). To address such attacks on FL-based applications in critical infrastructures, in this paper, we perform the first systematic study on the DP-exploited poisoning attacks from an adversarial point of view. We demonstrate that the DP method, despite providing a level of privacy guarantee, can effectively open a new poisoning attack vector for the adversary. Our theoretical analysis and empirical evaluation of a smart grid dataset show the FL performance degradation (sub-optimal model generation) scenario due to the differential noise-exploited selective model poisoning attacks. As a countermeasure, we propose a reinforcement learning-based differential privacy level selection (rDP) process. The rDP process utilizes the differential privacy parameters (privacy loss, information leakage probability, etc.) and the losses to intelligently generate an optimal privacy level for the nodes. The evaluation shows the accumulated reward and errors of the proposed technique converge to an optimal privacy policy.



## **45. Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL**

cs.LG

In the 10th International Conference on Learning Representations  (ICLR 2022)

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2106.05087v4)

**Authors**: Yanchao Sun, Ruijie Zheng, Yongyuan Liang, Furong Huang

**Abstracts**: Evaluating the worst-case performance of a reinforcement learning (RL) agent under the strongest/optimal adversarial perturbations on state observations (within some constraints) is crucial for understanding the robustness of RL agents. However, finding the optimal adversary is challenging, in terms of both whether we can find the optimal attack and how efficiently we can find it. Existing works on adversarial RL either use heuristics-based methods that may not find the strongest adversary, or directly train an RL-based adversary by treating the agent as a part of the environment, which can find the optimal adversary but may become intractable in a large state space. This paper introduces a novel attacking method to find the optimal attacks through collaboration between a designed function named "actor" and an RL-based learner named "director". The actor crafts state perturbations for a given policy perturbation direction, and the director learns to propose the best policy perturbation directions. Our proposed algorithm, PA-AD, is theoretically optimal and significantly more efficient than prior RL-based works in environments with large state spaces. Empirical results show that our proposed PA-AD universally outperforms state-of-the-art attacking methods in various Atari and MuJoCo environments. By applying PA-AD to adversarial training, we achieve state-of-the-art empirical robustness in multiple tasks under strong adversaries.



## **46. Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?**

cs.CV

Accepted at ICLR 2022

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2203.08392v2)

**Authors**: Yonggan Fu, Shunyao Zhang, Shang Wu, Cheng Wan, Yingyan Lin

**Abstracts**: Vision transformers (ViTs) have recently set off a new wave in neural architecture design thanks to their record-breaking performance in various vision tasks. In parallel, to fulfill the goal of deploying ViTs into real-world vision applications, their robustness against potential malicious attacks has gained increasing attention. In particular, recent works show that ViTs are more robust against adversarial attacks as compared with convolutional neural networks (CNNs), and conjecture that this is because ViTs focus more on capturing global interactions among different input/feature patches, leading to their improved robustness to local perturbations imposed by adversarial attacks. In this work, we ask an intriguing question: "Under what kinds of perturbations do ViTs become more vulnerable learners compared to CNNs?" Driven by this question, we first conduct a comprehensive experiment regarding the robustness of both ViTs and CNNs under various existing adversarial attacks to understand the underlying reason favoring their robustness. Based on the drawn insights, we then propose a dedicated attack framework, dubbed Patch-Fool, that fools the self-attention mechanism by attacking its basic component (i.e., a single patch) with a series of attention-aware optimization techniques. Interestingly, our Patch-Fool framework shows for the first time that ViTs are not necessarily more robust than CNNs against adversarial perturbations. In particular, we find that ViTs are more vulnerable learners compared with CNNs against our Patch-Fool attack which is consistent across extensive experiments, and the observations from Sparse/Mild Patch-Fool, two variants of Patch-Fool, indicate an intriguing insight that the perturbation density and strength on each patch seem to be the key factors that influence the robustness ranking between ViTs and CNNs.



## **47. Exploring Robust Architectures for Deep Artificial Neural Networks**

cs.LG

27 pages, 16 figures

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2106.15850v2)

**Authors**: Asim Waqas, Ghulam Rasool, Hamza Farooq, Nidhal C. Bouaynaya

**Abstracts**: The architectures of deep artificial neural networks (DANNs) are routinely studied to improve their predictive performance. However, the relationship between the architecture of a DANN and its robustness to noise and adversarial attacks is less explored. We investigate how the robustness of DANNs relates to their underlying graph architectures or structures. This study: (1) starts by exploring the design space of architectures of DANNs using graph-theoretic robustness measures; (2) transforms the graphs to DANN architectures to train/validate/test on various image classification tasks; (3) explores the relationship between the robustness of trained DANNs against noise and adversarial attacks and the robustness of their underlying architectures estimated via graph-theoretic measures. We show that the topological entropy and Olivier-Ricci curvature of the underlying graphs can quantify the robustness performance of DANNs. The said relationship is stronger for complex tasks and large DANNs. Our work will allow autoML and neural architecture search community to explore design spaces of robust and accurate DANNs.



## **48. User-Level Differential Privacy against Attribute Inference Attack of Speech Emotion Recognition in Federated Learning**

cs.CR

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.02500v1)

**Authors**: Tiantian Feng, Raghuveer Peri, Shrikanth Narayanan

**Abstracts**: Many existing privacy-enhanced speech emotion recognition (SER) frameworks focus on perturbing the original speech data through adversarial training within a centralized machine learning setup. However, this privacy protection scheme can fail since the adversary can still access the perturbed data. In recent years, distributed learning algorithms, especially federated learning (FL), have gained popularity to protect privacy in machine learning applications. While FL provides good intuition to safeguard privacy by keeping the data on local devices, prior work has shown that privacy attacks, such as attribute inference attacks, are achievable for SER systems trained using FL. In this work, we propose to evaluate the user-level differential privacy (UDP) in mitigating the privacy leaks of the SER system in FL. UDP provides theoretical privacy guarantees with privacy parameters $\epsilon$ and $\delta$. Our results show that the UDP can effectively decrease attribute information leakage while keeping the utility of the SER system with the adversary accessing one model update. However, the efficacy of the UDP suffers when the FL system leaks more model updates to the adversary. We make the code publicly available to reproduce the results in https://github.com/usc-sail/fed-ser-leakage.



## **49. Training-Free Robust Multimodal Learning via Sample-Wise Jacobian Regularization**

cs.CV

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.02485v1)

**Authors**: Zhengqi Gao, Sucheng Ren, Zihui Xue, Siting Li, Hang Zhao

**Abstracts**: Multimodal fusion emerges as an appealing technique to improve model performances on many tasks. Nevertheless, the robustness of such fusion methods is rarely involved in the present literature. In this paper, we propose a training-free robust late-fusion method by exploiting conditional independence assumption and Jacobian regularization. Our key is to minimize the Frobenius norm of a Jacobian matrix, where the resulting optimization problem is relaxed to a tractable Sylvester equation. Furthermore, we provide a theoretical error bound of our method and some insights about the function of the extra modality. Several numerical experiments on AV-MNIST, RAVDESS, and VGGsound demonstrate the efficacy of our method under both adversarial attacks and random corruptions.



## **50. Hear No Evil: Towards Adversarial Robustness of Automatic Speech Recognition via Multi-Task Learning**

eess.AS

Submitted to Insterspeech 2022

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.02381v1)

**Authors**: Nilaksh Das, Duen Horng Chau

**Abstracts**: As automatic speech recognition (ASR) systems are now being widely deployed in the wild, the increasing threat of adversarial attacks raises serious questions about the security and reliability of using such systems. On the other hand, multi-task learning (MTL) has shown success in training models that can resist adversarial attacks in the computer vision domain. In this work, we investigate the impact of performing such multi-task learning on the adversarial robustness of ASR models in the speech domain. We conduct extensive MTL experimentation by combining semantically diverse tasks such as accent classification and ASR, and evaluate a wide range of adversarial settings. Our thorough analysis reveals that performing MTL with semantically diverse tasks consistently makes it harder for an adversarial attack to succeed. We also discuss in detail the serious pitfalls and their related remedies that have a significant impact on the robustness of MTL models. Our proposed MTL approach shows considerable absolute improvements in adversarially targeted WER ranging from 17.25 up to 59.90 compared to single-task learning baselines (attention decoder and CTC respectively). Ours is the first in-depth study that uncovers adversarial robustness gains from multi-task learning for ASR.



