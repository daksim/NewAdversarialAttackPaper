# Latest Adversarial Attack Papers
**update at 2022-12-16 11:43:45**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Alternating Objectives Generates Stronger PGD-Based Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.07992v1) [paper-pdf](http://arxiv.org/pdf/2212.07992v1)

**Authors**: Nikolaos Antoniou, Efthymios Georgiou, Alexandros Potamianos

**Abstract**: Designing powerful adversarial attacks is of paramount importance for the evaluation of $\ell_p$-bounded adversarial defenses. Projected Gradient Descent (PGD) is one of the most effective and conceptually simple algorithms to generate such adversaries. The search space of PGD is dictated by the steepest ascent directions of an objective. Despite the plethora of objective function choices, there is no universally superior option and robustness overestimation may arise from ill-suited objective selection. Driven by this observation, we postulate that the combination of different objectives through a simple loss alternating scheme renders PGD more robust towards design choices. We experimentally verify this assertion on a synthetic-data example and by evaluating our proposed method across 25 different $\ell_{\infty}$-robust models and 3 datasets. The performance improvement is consistent, when compared to the single loss counterparts. In the CIFAR-10 dataset, our strongest adversarial attack outperforms all of the white-box components of AutoAttack (AA) ensemble, as well as the most powerful attacks existing on the literature, achieving state-of-the-art results in the computational budget of our study ($T=100$, no restarts).



## **2. A Feedback-optimization-based Model-free Attack Scheme in Networked Control Systems**

math.OC

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.07633v1) [paper-pdf](http://arxiv.org/pdf/2212.07633v1)

**Authors**: Xiaoyu Luo, Chongrong Fang, Jianping He, Chengcheng Zhao, Dario Paccagnan

**Abstract**: The data-driven attack strategies recently have received much attention when the system model is unknown to the adversary. Depending on the unknown linear system model, the existing data-driven attack strategies have already designed lots of efficient attack schemes. In this paper, we study a completely model-free attack scheme regardless of whether the unknown system model is linear or nonlinear. The objective of the adversary with limited capability is to compromise state variables such that the output value follows a malicious trajectory. Specifically, we first construct a zeroth-order feedback optimization framework and uninterruptedly use probing signals for real-time measurements. Then, we iteratively update the attack signals along the composite direction of the gradient estimates of the objective function evaluations and the projected gradients. These objective function evaluations can be obtained only by real-time measurements. Furthermore, we characterize the optimality of the proposed model-free attack via the optimality gap, which is affected by the dimensions of the attack signal, the iterations of solutions, and the convergence rate of the system. Finally, we extend the proposed attack scheme to the system with internal inherent noise and analyze the effects of noise on the optimality gap. Extensive simulations are conducted to show the effectiveness of the proposed attack scheme.



## **3. Dissecting Distribution Inference**

cs.LG

Accepted at SaTML 2023

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.07591v1) [paper-pdf](http://arxiv.org/pdf/2212.07591v1)

**Authors**: Anshuman Suri, Yifu Lu, Yanjin Chen, David Evans

**Abstract**: A distribution inference attack aims to infer statistical properties of data used to train machine learning models. These attacks are sometimes surprisingly potent, but the factors that impact distribution inference risk are not well understood and demonstrated attacks often rely on strong and unrealistic assumptions such as full knowledge of training environments even in supposedly black-box threat scenarios. To improve understanding of distribution inference risks, we develop a new black-box attack that even outperforms the best known white-box attack in most settings. Using this new attack, we evaluate distribution inference risk while relaxing a variety of assumptions about the adversary's knowledge under black-box access, like known model architectures and label-only access. Finally, we evaluate the effectiveness of previously proposed defenses and introduce new defenses. We find that although noise-based defenses appear to be ineffective, a simple re-sampling defense can be highly effective. Code is available at https://github.com/iamgroot42/dissecting_distribution_inference



## **4. SAIF: Sparse Adversarial and Interpretable Attack Framework**

cs.CV

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2212.07495v1) [paper-pdf](http://arxiv.org/pdf/2212.07495v1)

**Authors**: Tooba Imtiaz, Morgan Kohler, Jared Miller, Zifeng Wang, Mario Sznaier, Octavia Camps, Jennifer Dy

**Abstract**: Adversarial attacks hamper the decision-making ability of neural networks by perturbing the input signal. The addition of calculated small distortion to images, for instance, can deceive a well-trained image classification network. In this work, we propose a novel attack technique called Sparse Adversarial and Interpretable Attack Framework (SAIF). Specifically, we design imperceptible attacks that contain low-magnitude perturbations at a small number of pixels and leverage these sparse attacks to reveal the vulnerability of classifiers. We use the Frank-Wolfe (conditional gradient) algorithm to simultaneously optimize the attack perturbations for bounded magnitude and sparsity with $O(1/\sqrt{T})$ convergence. Empirical results show that SAIF computes highly imperceptible and interpretable adversarial examples, and outperforms state-of-the-art sparse attack methods on the ImageNet dataset.



## **5. XRand: Differentially Private Defense against Explanation-Guided Attacks**

cs.LG

To be published at AAAI 2023

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2212.04454v3) [paper-pdf](http://arxiv.org/pdf/2212.04454v3)

**Authors**: Truc Nguyen, Phung Lai, NhatHai Phan, My T. Thai

**Abstract**: Recent development in the field of explainable artificial intelligence (XAI) has helped improve trust in Machine-Learning-as-a-Service (MLaaS) systems, in which an explanation is provided together with the model prediction in response to each query. However, XAI also opens a door for adversaries to gain insights into the black-box models in MLaaS, thereby making the models more vulnerable to several attacks. For example, feature-based explanations (e.g., SHAP) could expose the top important features that a black-box model focuses on. Such disclosure has been exploited to craft effective backdoor triggers against malware classifiers. To address this trade-off, we introduce a new concept of achieving local differential privacy (LDP) in the explanations, and from that we establish a defense, called XRand, against such attacks. We show that our mechanism restricts the information that the adversary can learn about the top important features, while maintaining the faithfulness of the explanations.



## **6. The Devil is in the GAN: Backdoor Attacks and Defenses in Deep Generative Models**

cs.CR

17 pages, 11 figures, 3 tables

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2108.01644v2) [paper-pdf](http://arxiv.org/pdf/2108.01644v2)

**Authors**: Ambrish Rawat, Killian Levacher, Mathieu Sinn

**Abstract**: Deep Generative Models (DGMs) are a popular class of deep learning models which find widespread use because of their ability to synthesize data from complex, high-dimensional manifolds. However, even with their increasing industrial adoption, they haven't been subject to rigorous security and privacy analysis. In this work we examine one such aspect, namely backdoor attacks on DGMs which can significantly limit the applicability of pre-trained models within a model supply chain and at the very least cause massive reputation damage for companies outsourcing DGMs form third parties.   While similar attacks scenarios have been studied in the context of classical prediction models, their manifestation in DGMs hasn't received the same attention. To this end we propose novel training-time attacks which result in corrupted DGMs that synthesize regular data under normal operations and designated target outputs for inputs sampled from a trigger distribution. These attacks are based on an adversarial loss function that combines the dual objectives of attack stealth and fidelity. We systematically analyze these attacks, and show their effectiveness for a variety of approaches like Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), as well as different data domains including images and audio. Our experiments show that - even for large-scale industry-grade DGMs (like StyleGAN) - our attacks can be mounted with only modest computational effort. We also motivate suitable defenses based on static/dynamic model and output inspections, demonstrate their usefulness, and prescribe a practical and comprehensive defense strategy that paves the way for safe usage of DGMs.



## **7. Object-fabrication Targeted Attack for Object Detection**

cs.CV

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2212.06431v2) [paper-pdf](http://arxiv.org/pdf/2212.06431v2)

**Authors**: Xuchong Zhang, Changfeng Sun, Haoliang Han, Hang Wang, Hongbin Sun, Nanning Zheng

**Abstract**: Recent researches show that the deep learning based object detection is vulnerable to adversarial examples. Generally, the adversarial attack for object detection contains targeted attack and untargeted attack. According to our detailed investigations, the research on the former is relatively fewer than the latter and all the existing methods for the targeted attack follow the same mode, i.e., the object-mislabeling mode that misleads detectors to mislabel the detected object as a specific wrong label. However, this mode has limited attack success rate, universal and generalization performances. In this paper, we propose a new object-fabrication targeted attack mode which can mislead detectors to `fabricate' extra false objects with specific target labels. Furthermore, we design a dual attention based targeted feature space attack method to implement the proposed targeted attack mode. The attack performances of the proposed mode and method are evaluated on MS COCO and BDD100K datasets using FasterRCNN and YOLOv5. Evaluation results demonstrate that, the proposed object-fabrication targeted attack mode and the corresponding targeted feature space attack method show significant improvements in terms of image-specific attack, universal performance and generalization capability, compared with the previous targeted attack for object detection. Code will be made available.



## **8. ARCADE: Adversarially Regularized Convolutional Autoencoder for Network Anomaly Detection**

cs.LG

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2205.01432v3) [paper-pdf](http://arxiv.org/pdf/2205.01432v3)

**Authors**: Willian T. Lunardi, Martin Andreoni Lopez, Jean-Pierre Giacalone

**Abstract**: As the number of heterogenous IP-connected devices and traffic volume increase, so does the potential for security breaches. The undetected exploitation of these breaches can bring severe cybersecurity and privacy risks. Anomaly-based \acp{IDS} play an essential role in network security. In this paper, we present a practical unsupervised anomaly-based deep learning detection system called ARCADE (Adversarially Regularized Convolutional Autoencoder for unsupervised network anomaly DEtection). With a convolutional \ac{AE}, ARCADE automatically builds a profile of the normal traffic using a subset of raw bytes of a few initial packets of network flows so that potential network anomalies and intrusions can be efficiently detected before they cause more damage to the network. ARCADE is trained exclusively on normal traffic. An adversarial training strategy is proposed to regularize and decrease the \ac{AE}'s capabilities to reconstruct network flows that are out-of-the-normal distribution, thereby improving its anomaly detection capabilities. The proposed approach is more effective than state-of-the-art deep learning approaches for network anomaly detection. Even when examining only two initial packets of a network flow, ARCADE can effectively detect malware infection and network attacks. ARCADE presents 20 times fewer parameters than baselines, achieving significantly faster detection speed and reaction time.



## **9. The Quantum Chernoff Divergence in Advantage Distillation for QKD and DIQKD**

quant-ph

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2212.06975v1) [paper-pdf](http://arxiv.org/pdf/2212.06975v1)

**Authors**: Mikka Stasiuk, Norbert Lutkenhaus, Ernest Y. -Z. Tan

**Abstract**: Device-independent quantum key distribution (DIQKD) aims to mitigate adversarial exploitation of imperfections in quantum devices, by providing an approach for secret key distillation with modest security assumptions. Advantage distillation, a two-way communication procedure in error correction, has proven effective in raising noise tolerances in both device-dependent and device-independent QKD. Previously, device-independent security proofs against IID collective attacks were developed for an advantage distillation protocol known as the repetition-code protocol, based on security conditions involving the fidelity between some states in the protocol. However, there exists a gap between the sufficient and necessary security conditions, which hinders the calculation of tight noise-tolerance bounds based on the fidelity. We close this gap by presenting an alternative proof structure that replaces the fidelity with the quantum Chernoff divergence, a distinguishability measure that arises in symmetric hypothesis testing. Working in the IID collective attacks model, we derive matching sufficient and necessary conditions for the repetition-code protocol to be secure (up to a natural conjecture regarding the latter case) in terms of the quantum Chernoff divergence, hence indicating that this serves as the relevant quantity of interest for this protocol. Furthermore, using this security condition we obtain some improvements over previous results on the noise tolerance thresholds for DIQKD. Our results provide insight into a fundamental question in quantum information theory regarding the circumstances under which DIQKD is possible



## **10. Adversarial Attacks and Defences for Skin Cancer Classification**

cs.CV

6 pages, 7 figures, 2 tables, 2nd International Conference for  Advancement in Technology (ICONAT 2023), Goa, India

**SubmitDate**: 2022-12-13    [abs](http://arxiv.org/abs/2212.06822v1) [paper-pdf](http://arxiv.org/pdf/2212.06822v1)

**Authors**: Vinay Jogani, Joy Purohit, Ishaan Shivhare, Samina Attari, Shraddha Surtkar

**Abstract**: There has been a concurrent significant improvement in the medical images used to facilitate diagnosis and the performance of machine learning techniques to perform tasks such as classification, detection, and segmentation in recent years. As a result, a rapid increase in the usage of such systems can be observed in the healthcare industry, for instance in the form of medical image classification systems, where these models have achieved diagnostic parity with human physicians. One such application where this can be observed is in computer vision tasks such as the classification of skin lesions in dermatoscopic images. However, as stakeholders in the healthcare industry, such as insurance companies, continue to invest extensively in machine learning infrastructure, it becomes increasingly important to understand the vulnerabilities in such systems. Due to the highly critical nature of the tasks being carried out by these machine learning models, it is necessary to analyze techniques that could be used to take advantage of these vulnerabilities and methods to defend against them. This paper explores common adversarial attack techniques. The Fast Sign Gradient Method and Projected Descent Gradient are used against a Convolutional Neural Network trained to classify dermatoscopic images of skin lesions. Following that, it also discusses one of the most popular adversarial defense techniques, adversarial training. The performance of the model that has been trained on adversarial examples is then tested against the previously mentioned attacks, and recommendations to improve neural networks robustness are thus provided based on the results of the experiment.



## **11. Towards Efficient and Domain-Agnostic Evasion Attack with High-dimensional Categorical Inputs**

cs.LG

AAAI 2023

**SubmitDate**: 2022-12-13    [abs](http://arxiv.org/abs/2212.06836v1) [paper-pdf](http://arxiv.org/pdf/2212.06836v1)

**Authors**: Hongyan Bao, Yufei Han, Yujun Zhou, Xin Gao, Xiangliang Zhang

**Abstract**: Our work targets at searching feasible adversarial perturbation to attack a classifier with high-dimensional categorical inputs in a domain-agnostic setting. This is intrinsically an NP-hard knapsack problem where the exploration space becomes explosively larger as the feature dimension increases. Without the help of domain knowledge, solving this problem via heuristic method, such as Branch-and-Bound, suffers from exponential complexity, yet can bring arbitrarily bad attack results. We address the challenge via the lens of multi-armed bandit based combinatorial search. Our proposed method, namely FEAT, treats modifying each categorical feature as pulling an arm in multi-armed bandit programming. Our objective is to achieve highly efficient and effective attack using an Orthogonal Matching Pursuit (OMP)-enhanced Upper Confidence Bound (UCB) exploration strategy. Our theoretical analysis bounding the regret gap of FEAT guarantees its practical attack performance. In empirical analysis, we compare FEAT with other state-of-the-art domain-agnostic attack methods over various real-world categorical data sets of different applications. Substantial experimental observations confirm the expected efficiency and attack effectiveness of FEAT applied in different application scenarios. Our work further hints the applicability of FEAT for assessing the adversarial vulnerability of classification systems with high-dimensional categorical inputs.



## **12. The Importance of Image Interpretation: Patterns of Semantic Misclassification in Real-World Adversarial Images**

cs.CV

International Conference on Multimedia Modeling (MMM) 2023. Resources  are publicly available at  https://github.com/ZhengyuZhao/Targeted-Transfer/tree/main/human_eval

**SubmitDate**: 2022-12-13    [abs](http://arxiv.org/abs/2206.01467v2) [paper-pdf](http://arxiv.org/pdf/2206.01467v2)

**Authors**: Zhengyu Zhao, Nga Dang, Martha Larson

**Abstract**: Adversarial images are created with the intention of causing an image classifier to produce a misclassification. In this paper, we propose that adversarial images should be evaluated based on semantic mismatch, rather than label mismatch, as used in current work. In other words, we propose that an image of a "mug" would be considered adversarial if classified as "turnip", but not as "cup", as current systems would assume. Our novel idea of taking semantic misclassification into account in the evaluation of adversarial images offers two benefits. First, it is a more realistic conceptualization of what makes an image adversarial, which is important in order to fully understand the implications of adversarial images for security and privacy. Second, it makes it possible to evaluate the transferability of adversarial images to a real-world classifier, without requiring the classifier's label set to have been available during the creation of the images. The paper carries out an evaluation of a transfer attack on a real-world image classifier that is made possible by our semantic misclassification approach. The attack reveals patterns in the semantics of adversarial misclassifications that could not be investigated using conventional label mismatch.



## **13. Unfolding Local Growth Rate Estimates for (Almost) Perfect Adversarial Detection**

cs.CV

accepted at VISAPP23

**SubmitDate**: 2022-12-13    [abs](http://arxiv.org/abs/2212.06776v1) [paper-pdf](http://arxiv.org/pdf/2212.06776v1)

**Authors**: Peter Lorenz, Margret Keuper, Janis Keuper

**Abstract**: Convolutional neural networks (CNN) define the state-of-the-art solution on many perceptual tasks. However, current CNN approaches largely remain vulnerable against adversarial perturbations of the input that have been crafted specifically to fool the system while being quasi-imperceptible to the human eye. In recent years, various approaches have been proposed to defend CNNs against such attacks, for example by model hardening or by adding explicit defence mechanisms. Thereby, a small "detector" is included in the network and trained on the binary classification task of distinguishing genuine data from data containing adversarial perturbations. In this work, we propose a simple and light-weight detector, which leverages recent findings on the relation between networks' local intrinsic dimensionality (LID) and adversarial attacks. Based on a re-interpretation of the LID measure and several simple adaptations, we surpass the state-of-the-art on adversarial detection by a significant margin and reach almost perfect results in terms of F1-score for several networks and datasets. Sources available at: https://github.com/adverML/multiLID



## **14. Adversarial Detection by Approximation of Ensemble Boundary**

cs.LG

6 pages, 3 figures, 5 tables

**SubmitDate**: 2022-12-13    [abs](http://arxiv.org/abs/2211.10227v3) [paper-pdf](http://arxiv.org/pdf/2211.10227v3)

**Authors**: T. Windeatt

**Abstract**: A spectral approximation of a Boolean function is proposed for approximating the decision boundary of an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The Walsh combination of relatively weak DNN classifiers is shown experimentally to be capable of detecting adversarial attacks. By observing the difference in Walsh coefficient approximation between clean and adversarial images, it appears that transferability of attack may be used for detection. Approximating the decision boundary may also aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class ensemble decision boundaries could in principle be applied to any application area. Code for this paper implementing Walsh Coefficient Examples of approximating artificial Boolean functions can be found at https://doi.org/10.24433/CO.3695905.v1



## **15. Pixel is All You Need: Adversarial Trajectory-Ensemble Active Learning for Salient Object Detection**

cs.CV

9 pages, 8 figures

**SubmitDate**: 2022-12-13    [abs](http://arxiv.org/abs/2212.06493v1) [paper-pdf](http://arxiv.org/pdf/2212.06493v1)

**Authors**: Zhenyu Wu, Lin Wang, Wei Wang, Qing Xia, Chenglizhao Chen, Aimin Hao, Shuo Li

**Abstract**: Although weakly-supervised techniques can reduce the labeling effort, it is unclear whether a saliency model trained with weakly-supervised data (e.g., point annotation) can achieve the equivalent performance of its fully-supervised version. This paper attempts to answer this unexplored question by proving a hypothesis: there is a point-labeled dataset where saliency models trained on it can achieve equivalent performance when trained on the densely annotated dataset. To prove this conjecture, we proposed a novel yet effective adversarial trajectory-ensemble active learning (ATAL). Our contributions are three-fold: 1) Our proposed adversarial attack triggering uncertainty can conquer the overconfidence of existing active learning methods and accurately locate these uncertain pixels. {2)} Our proposed trajectory-ensemble uncertainty estimation method maintains the advantages of the ensemble networks while significantly reducing the computational cost. {3)} Our proposed relationship-aware diversity sampling algorithm can conquer oversampling while boosting performance. Experimental results show that our ATAL can find such a point-labeled dataset, where a saliency model trained on it obtained $97\%$ -- $99\%$ performance of its fully-supervised version with only ten annotated points per image.



## **16. Adversarially Robust Video Perception by Seeing Motion**

cs.CV

**SubmitDate**: 2022-12-13    [abs](http://arxiv.org/abs/2212.07815v1) [paper-pdf](http://arxiv.org/pdf/2212.07815v1)

**Authors**: Lingyu Zhang, Chengzhi Mao, Junfeng Yang, Carl Vondrick

**Abstract**: Despite their excellent performance, state-of-the-art computer vision models often fail when they encounter adversarial examples. Video perception models tend to be more fragile under attacks, because the adversary has more places to manipulate in high-dimensional data. In this paper, we find one reason for video models' vulnerability is that they fail to perceive the correct motion under adversarial perturbations. Inspired by the extensive evidence that motion is a key factor for the human visual system, we propose to correct what the model sees by restoring the perceived motion information. Since motion information is an intrinsic structure of the video data, recovering motion signals can be done at inference time without any human annotation, which allows the model to adapt to unforeseen, worst-case inputs. Visualizations and empirical experiments on UCF-101 and HMDB-51 datasets show that restoring motion information in deep vision models improves adversarial robustness. Even under adaptive attacks where the adversary knows our defense, our algorithm is still effective. Our work provides new insight into robust video perception algorithms by using intrinsic structures from the data. Our webpage is available at https://motion4robust.cs.columbia.edu.



## **17. Securing the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples**

cs.NE

**SubmitDate**: 2022-12-12    [abs](http://arxiv.org/abs/2209.03358v2) [paper-pdf](http://arxiv.org/pdf/2209.03358v2)

**Authors**: Nuo Xu, Kaleel Mahmood, Haowen Fang, Ethan Rathbun, Caiwen Ding, Wujie Wen

**Abstract**: Spiking neural networks (SNNs) have attracted much attention for their high energy efficiency and for recent advances in their classification performance. However, unlike traditional deep learning approaches, the analysis and study of the robustness of SNNs to adversarial examples remain relatively underdeveloped. In this work we focus on advancing the adversarial attack side of SNNs and make three major contributions. First, we show that successful white-box adversarial attacks on SNNs are highly dependent on the underlying surrogate gradient technique. Second, using the best surrogate gradient technique, we analyze the transferability of adversarial attacks on SNNs and other state-of-the-art architectures like Vision Transformers (ViTs) and Big Transfer Convolutional Neural Networks (CNNs). We demonstrate that SNNs are not often deceived by adversarial examples generated by Vision Transformers and certain types of CNNs. Third, due to the lack of an ubiquitous white-box attack that is effective across both the SNN and CNN/ViT domains, we develop a new white-box attack, the Auto Self-Attention Gradient Attack (Auto SAGA). Our novel attack generates adversarial examples capable of fooling both SNN models and non-SNN models simultaneously. Auto SAGA is as much as $87.9\%$ more effective on SNN/ViT model ensembles than conventional white-box attacks like PGD. Our experiments and analyses are broad and rigorous covering three datasets (CIFAR-10, CIFAR-100 and ImageNet), five different white-box attacks and nineteen different classifier models (seven for each CIFAR dataset and five different models for ImageNet).



## **18. Increasing the Cost of Model Extraction with Calibrated Proof of Work**

cs.CR

Published as a conference paper at ICLR 2022 (Spotlight - 5% of  submitted papers)

**SubmitDate**: 2022-12-12    [abs](http://arxiv.org/abs/2201.09243v3) [paper-pdf](http://arxiv.org/pdf/2201.09243v3)

**Authors**: Adam Dziedzic, Muhammad Ahmad Kaleem, Yu Shen Lu, Nicolas Papernot

**Abstract**: In model extraction attacks, adversaries can steal a machine learning model exposed via a public API by repeatedly querying it and adjusting their own model based on obtained predictions. To prevent model stealing, existing defenses focus on detecting malicious queries, truncating, or distorting outputs, thus necessarily introducing a tradeoff between robustness and model utility for legitimate users. Instead, we propose to impede model extraction by requiring users to complete a proof-of-work before they can read the model's predictions. This deters attackers by greatly increasing (even up to 100x) the computational effort needed to leverage query access for model extraction. Since we calibrate the effort required to complete the proof-of-work to each query, this only introduces a slight overhead for regular users (up to 2x). To achieve this, our calibration applies tools from differential privacy to measure the information revealed by a query. Our method requires no modification of the victim model and can be applied by machine learning practitioners to guard their publicly exposed models against being easily stolen.



## **19. Dictionary Attacks on Speaker Verification**

cs.SD

Accepted in IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2022-12-12    [abs](http://arxiv.org/abs/2204.11304v2) [paper-pdf](http://arxiv.org/pdf/2204.11304v2)

**Authors**: Mirko Marras, Pawel Korus, Anubhav Jain, Nasir Memon

**Abstract**: In this paper, we propose dictionary attacks against speaker verification - a novel attack vector that aims to match a large fraction of speaker population by chance. We introduce a generic formulation of the attack that can be used with various speech representations and threat models. The attacker uses adversarial optimization to maximize raw similarity of speaker embeddings between a seed speech sample and a proxy population. The resulting master voice successfully matches a non-trivial fraction of people in an unknown population. Adversarial waveforms obtained with our approach can match on average 69% of females and 38% of males enrolled in the target system at a strict decision threshold calibrated to yield false alarm rate of 1%. By using the attack with a black-box voice cloning system, we obtain master voices that are effective in the most challenging conditions and transferable between speaker encoders. We also show that, combined with multiple attempts, this attack opens even more to serious issues on the security of these systems.



## **20. Reversing Skin Cancer Adversarial Examples by Multiscale Diffusive and Denoising Aggregation Mechanism**

cs.CV

11 pages

**SubmitDate**: 2022-12-12    [abs](http://arxiv.org/abs/2208.10373v2) [paper-pdf](http://arxiv.org/pdf/2208.10373v2)

**Authors**: Yongwei Wang, Yuan Li, Zhiqi Shen

**Abstract**: Reliable skin cancer diagnosis models play an essential role in early screening and medical intervention. Prevailing computer-aided skin cancer classification systems employ deep learning approaches. However, recent studies reveal their extreme vulnerability to adversarial attacks -- often imperceptible perturbations to significantly reduce the performances of skin cancer diagnosis models. To mitigate these threats, this work presents a simple, effective, and resource-efficient defense framework by reverse engineering adversarial perturbations in skin cancer images. Specifically, a multiscale image pyramid is first established to better preserve discriminative structures in the medical imaging domain. To neutralize adversarial effects, skin images at different scales are then progressively diffused by injecting isotropic Gaussian noises to move the adversarial examples to the clean image manifold. Crucially, to further reverse adversarial noises and suppress redundant injected noises, a novel multiscale denoising mechanism is carefully designed that aggregates image information from neighboring scales. We evaluated the defensive effectiveness of our method on ISIC 2019, a largest skin cancer multiclass classification dataset. Experimental results demonstrate that the proposed method can successfully reverse adversarial perturbations from different attacks and significantly outperform some state-of-the-art methods in defending skin cancer diagnosis models.



## **21. HOTCOLD Block: Fooling Thermal Infrared Detectors with a Novel Wearable Design**

cs.CV

Accepted to AAAI 2023

**SubmitDate**: 2022-12-12    [abs](http://arxiv.org/abs/2212.05709v1) [paper-pdf](http://arxiv.org/pdf/2212.05709v1)

**Authors**: Hui Wei, Zhixiang Wang, Xuemei Jia, Yinqiang Zheng, Hao Tang, Shin'ichi Satoh, Zheng Wang

**Abstract**: Adversarial attacks on thermal infrared imaging expose the risk of related applications. Estimating the security of these systems is essential for safely deploying them in the real world. In many cases, realizing the attacks in the physical space requires elaborate special perturbations. These solutions are often \emph{impractical} and \emph{attention-grabbing}. To address the need for a physically practical and stealthy adversarial attack, we introduce \textsc{HotCold} Block, a novel physical attack for infrared detectors that hide persons utilizing the wearable Warming Paste and Cooling Paste. By attaching these readily available temperature-controlled materials to the body, \textsc{HotCold} Block evades human eyes efficiently. Moreover, unlike existing methods that build adversarial patches with complex texture and structure features, \textsc{HotCold} Block utilizes an SSP-oriented adversarial optimization algorithm that enables attacks with pure color blocks and explores the influence of size, shape, and position on attack performance. Extensive experimental results in both digital and physical environments demonstrate the performance of our proposed \textsc{HotCold} Block. \emph{Code is available: \textcolor{magenta}{https://github.com/weihui1308/HOTCOLDBlock}}.



## **22. REAP: A Large-Scale Realistic Adversarial Patch Benchmark**

cs.CV

Code and benchmark can be found at  https://github.com/wagner-group/reap-benchmark

**SubmitDate**: 2022-12-12    [abs](http://arxiv.org/abs/2212.05680v1) [paper-pdf](http://arxiv.org/pdf/2212.05680v1)

**Authors**: Nabeel Hingun, Chawin Sitawarin, Jerry Li, David Wagner

**Abstract**: Machine learning models are known to be susceptible to adversarial perturbation. One famous attack is the adversarial patch, a sticker with a particularly crafted pattern that makes the model incorrectly predict the object it is placed on. This attack presents a critical threat to cyber-physical systems that rely on cameras such as autonomous cars. Despite the significance of the problem, conducting research in this setting has been difficult; evaluating attacks and defenses in the real world is exceptionally costly while synthetic data are unrealistic. In this work, we propose the REAP (REalistic Adversarial Patch) benchmark, a digital benchmark that allows the user to evaluate patch attacks on real images, and under real-world conditions. Built on top of the Mapillary Vistas dataset, our benchmark contains over 14,000 traffic signs. Each sign is augmented with a pair of geometric and lighting transformations, which can be used to apply a digitally generated patch realistically onto the sign. Using our benchmark, we perform the first large-scale assessments of adversarial patch attacks under realistic conditions. Our experiments suggest that adversarial patch attacks may present a smaller threat than previously believed and that the success rate of an attack on simpler digital simulations is not predictive of its actual effectiveness in practice. We release our benchmark publicly at https://github.com/wagner-group/reap-benchmark.



## **23. DISCO: Adversarial Defense with Local Implicit Functions**

cs.CV

Accepted to Neurips 2022

**SubmitDate**: 2022-12-11    [abs](http://arxiv.org/abs/2212.05630v1) [paper-pdf](http://arxiv.org/pdf/2212.05630v1)

**Authors**: Chih-Hui Ho, Nuno Vasconcelos

**Abstract**: The problem of adversarial defenses for image classification, where the goal is to robustify a classifier against adversarial examples, is considered. Inspired by the hypothesis that these examples lie beyond the natural image manifold, a novel aDversarIal defenSe with local impliCit functiOns (DISCO) is proposed to remove adversarial perturbations by localized manifold projections. DISCO consumes an adversarial image and a query pixel location and outputs a clean RGB value at the location. It is implemented with an encoder and a local implicit module, where the former produces per-pixel deep features and the latter uses the features in the neighborhood of query pixel for predicting the clean RGB value. Extensive experiments demonstrate that both DISCO and its cascade version outperform prior defenses, regardless of whether the defense is known to the attacker. DISCO is also shown to be data and parameter efficient and to mount defenses that transfers across datasets, classifiers and attacks.



## **24. On Deep Learning in Password Guessing, a Survey**

cs.CR

8 pages, 4 figures, 3 tables. arXiv admin note: substantial text  overlap with arXiv:2208.06943

**SubmitDate**: 2022-12-11    [abs](http://arxiv.org/abs/2208.10413v2) [paper-pdf](http://arxiv.org/pdf/2208.10413v2)

**Authors**: Fangyi Yu

**Abstract**: The security of passwords is dependent on a thorough understanding of the strategies used by attackers. Unfortunately, real-world adversaries use pragmatic guessing tactics like dictionary attacks, which are difficult to simulate in password security research. Dictionary attacks must be carefully configured and modified to be representative of the actual threat. This approach, however, needs domain-specific knowledge and expertise that are difficult to duplicate. This paper compares various deep learning-based password guessing approaches that do not require domain knowledge or assumptions about users' password structures and combinations. The involved model categories are Recurrent Neural Networks, Generative Adversarial Networks, Autoencoder, and Attention mechanisms. Additionally, we proposed a promising research experimental design on using variations of IWGAN on password guessing under non-targeted offline attacks. Using these advanced strategies, we can enhance password security and create more accurate and efficient Password Strength Meters.



## **25. Security Defense of Large Scale Networks Under False Data Injection Attacks: An Attack Detection Scheduling Approach**

eess.SY

**SubmitDate**: 2022-12-11    [abs](http://arxiv.org/abs/2212.05500v1) [paper-pdf](http://arxiv.org/pdf/2212.05500v1)

**Authors**: Yuhan Suo, Senchun Chai, Runqi Chai, Zhong-Hua Pang, Yuanqing Xia, Guo-Ping Liu

**Abstract**: In large scale networks, communication links between nodes are easily injected with false data by adversaries, so this paper proposes a novel security defense strategy to ensure the security of the network from the perspective of attack detection scheduling. Compared with existing attack detection methods, the attack detection scheduling strategy in this paper only needs to detect half of the neighbor node information to ensure the security of the node local state estimation. We first formulate the problem of selecting the sensor to be detected as a combinatorial optimization problem, which is Nondeterminism Polynomial hard (NP-hard). To solve the above problem, we convert the objective function into a submodular function. Then, we propose an attack detection scheduling algorithm based on sequential submodular maximization, which incorporates expert problem to better cope with dynamic attack strategies. The proposed algorithm can run in polynomial time with a theoretical lower bound on the optimization rate. In addition, the proposed algorithm can guarantee the security of the whole network under two kinds of insecurity conditions from the perspective of the augmented estimation error. Finally, a numerical simulation of the industrial continuous stirred tank reactor verifies the effectiveness of the developed approach.



## **26. General Adversarial Defense Against Black-box Attacks via Pixel Level and Feature Level Distribution Alignments**

cs.CV

**SubmitDate**: 2022-12-11    [abs](http://arxiv.org/abs/2212.05387v1) [paper-pdf](http://arxiv.org/pdf/2212.05387v1)

**Authors**: Xiaogang Xu, Hengshuang Zhao, Philip Torr, Jiaya Jia

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to the black-box adversarial attack that is highly transferable. This threat comes from the distribution gap between adversarial and clean samples in feature space of the target DNNs. In this paper, we use Deep Generative Networks (DGNs) with a novel training mechanism to eliminate the distribution gap. The trained DGNs align the distribution of adversarial samples with clean ones for the target DNNs by translating pixel values. Different from previous work, we propose a more effective pixel level training constraint to make this achievable, thus enhancing robustness on adversarial samples. Further, a class-aware feature-level constraint is formulated for integrated distribution alignment. Our approach is general and applicable to multiple tasks, including image classification, semantic segmentation, and object detection. We conduct extensive experiments on different datasets. Our strategy demonstrates its unique effectiveness and generality against black-box attacks.



## **27. Mitigating Adversarial Gray-Box Attacks Against Phishing Detectors**

cs.CR

**SubmitDate**: 2022-12-11    [abs](http://arxiv.org/abs/2212.05380v1) [paper-pdf](http://arxiv.org/pdf/2212.05380v1)

**Authors**: Giovanni Apruzzese, V. S. Subrahmanian

**Abstract**: Although machine learning based algorithms have been extensively used for detecting phishing websites, there has been relatively little work on how adversaries may attack such "phishing detectors" (PDs for short). In this paper, we propose a set of Gray-Box attacks on PDs that an adversary may use which vary depending on the knowledge that he has about the PD. We show that these attacks severely degrade the effectiveness of several existing PDs. We then propose the concept of operation chains that iteratively map an original set of features to a new set of features and develop the "Protective Operation Chain" (POC for short) algorithm. POC leverages the combination of random feature selection and feature mappings in order to increase the attacker's uncertainty about the target PD. Using 3 existing publicly available datasets plus a fourth that we have created and will release upon the publication of this paper, we show that POC is more robust to these attacks than past competing work, while preserving predictive performance when no adversarial attacks are present. Moreover, POC is robust to attacks on 13 different classifiers, not just one. These results are shown to be statistically significant at the p < 0.001 level.



## **28. Targeted Adversarial Attacks on Deep Reinforcement Learning Policies via Model Checking**

cs.LG

ICAART 2023 Paper (Technical Report)

**SubmitDate**: 2022-12-10    [abs](http://arxiv.org/abs/2212.05337v1) [paper-pdf](http://arxiv.org/pdf/2212.05337v1)

**Authors**: Dennis Gross, Thiago D. Simao, Nils Jansen, Guillermo A. Perez

**Abstract**: Deep Reinforcement Learning (RL) agents are susceptible to adversarial noise in their observations that can mislead their policies and decrease their performance. However, an adversary may be interested not only in decreasing the reward, but also in modifying specific temporal logic properties of the policy. This paper presents a metric that measures the exact impact of adversarial attacks against such properties. We use this metric to craft optimal adversarial attacks. Furthermore, we introduce a model checking method that allows us to verify the robustness of RL policies against adversarial attacks. Our empirical analysis confirms (1) the quality of our metric to craft adversarial attacks against temporal logic properties, and (2) that we are able to concisely assess a system's robustness against attacks.



## **29. UPTON: Unattributable Authorship Text via Data Poisoning**

cs.CY

**SubmitDate**: 2022-12-10    [abs](http://arxiv.org/abs/2211.09717v2) [paper-pdf](http://arxiv.org/pdf/2211.09717v2)

**Authors**: Ziyao Wang, Thai Le, Dongwon Lee

**Abstract**: In online medium such as opinion column in Bloomberg, The Guardian and Western Journal, aspiring writers post their writings for various reasons with their names often proudly open. However, it may occur that such a writer wants to write in other venues anonymously or under a pseudonym (e.g., activist, whistle-blower). However, if an attacker has already built an accurate authorship attribution (AA) model based off of the writings from such platforms, attributing an anonymous writing to the known authorship is possible. Therefore, in this work, we ask a question "can one make the writings and texts, T, in the open spaces such as opinion sharing platforms unattributable so that AA models trained from T cannot attribute authorship well?" Toward this question, we present a novel solution, UPTON, that exploits textual data poisoning method to disturb the training process of AA models. UPTON uses data poisoning to destroy the authorship feature only in training samples by perturbing them, and try to make released textual data unlearnable on deep neuron networks. It is different from previous obfuscation works, that use adversarial attack to modify the test samples and mislead an AA model, and also the backdoor works, which use trigger words both in test and training samples and only change the model output when trigger words occur. Using four authorship datasets (e.g., IMDb10, IMDb64, Enron and WJO), then, we present empirical validation where: (1)UPTON is able to downgrade the test accuracy to about 30% with carefully designed target-selection methods. (2)UPTON poisoning is able to preserve most of the original semantics. The BERTSCORE between the clean and UPTON poisoned texts are higher than 0.95. The number is very closed to 1.00, which means no sematic change. (3)UPTON is also robust towards spelling correction systems.



## **30. Understanding and Combating Robust Overfitting via Input Loss Landscape Analysis and Regularization**

cs.LG

published in journal Pattern Recognition:  https://www.sciencedirect.com/science/article/pii/S0031320322007087?via%3Dihub

**SubmitDate**: 2022-12-09    [abs](http://arxiv.org/abs/2212.04985v1) [paper-pdf](http://arxiv.org/pdf/2212.04985v1)

**Authors**: Lin Li, Michael Spratling

**Abstract**: Adversarial training is widely used to improve the robustness of deep neural networks to adversarial attack. However, adversarial training is prone to overfitting, and the cause is far from clear. This work sheds light on the mechanisms underlying overfitting through analyzing the loss landscape w.r.t. the input. We find that robust overfitting results from standard training, specifically the minimization of the clean loss, and can be mitigated by regularization of the loss gradients. Moreover, we find that robust overfitting turns severer during adversarial training partially because the gradient regularization effect of adversarial training becomes weaker due to the increase in the loss landscapes curvature. To improve robust generalization, we propose a new regularizer to smooth the loss landscape by penalizing the weighted logits variation along the adversarial direction. Our method significantly mitigates robust overfitting and achieves the highest robustness and efficiency compared to similar previous methods. Code is available at https://github.com/TreeLLi/Combating-RO-AdvLC.



## **31. Expeditious Saliency-guided Mix-up through Random Gradient Thresholding**

cs.CV

Accepted Long paper at 2nd Practical-DL Workshop at AAAI 2023

**SubmitDate**: 2022-12-09    [abs](http://arxiv.org/abs/2212.04875v1) [paper-pdf](http://arxiv.org/pdf/2212.04875v1)

**Authors**: Minh-Long Luu, Zeyi Huang, Eric P. Xing, Yong Jae Lee, Haohan Wang

**Abstract**: Mix-up training approaches have proven to be effective in improving the generalization ability of Deep Neural Networks. Over the years, the research community expands mix-up methods into two directions, with extensive efforts to improve saliency-guided procedures but minimal focus on the arbitrary path, leaving the randomization domain unexplored. In this paper, inspired by the superior qualities of each direction over one another, we introduce a novel method that lies at the junction of the two routes. By combining the best elements of randomness and saliency utilization, our method balances speed, simplicity, and accuracy. We name our method R-Mix following the concept of "Random Mix-up". We demonstrate its effectiveness in generalization, weakly supervised object localization, calibration, and robustness to adversarial attacks. Finally, in order to address the question of whether there exists a better decision protocol, we train a Reinforcement Learning agent that decides the mix-up policies based on the classifier's performance, reducing dependency on human-designed objectives and hyperparameter tuning. Extensive experiments further show that the agent is capable of performing at the cutting-edge level, laying the foundation for a fully automatic mix-up. Our code is released at [https://github.com/minhlong94/Random-Mixup].



## **32. Unfooling Perturbation-Based Post Hoc Explainers**

cs.AI

Accepted to AAAI-23. 9 pages (not including references and  supplemental)

**SubmitDate**: 2022-12-09    [abs](http://arxiv.org/abs/2205.14772v2) [paper-pdf](http://arxiv.org/pdf/2205.14772v2)

**Authors**: Zachariah Carmichael, Walter J Scheirer

**Abstract**: Monumental advancements in artificial intelligence (AI) have lured the interest of doctors, lenders, judges, and other professionals. While these high-stakes decision-makers are optimistic about the technology, those familiar with AI systems are wary about the lack of transparency of its decision-making processes. Perturbation-based post hoc explainers offer a model agnostic means of interpreting these systems while only requiring query-level access. However, recent work demonstrates that these explainers can be fooled adversarially. This discovery has adverse implications for auditors, regulators, and other sentinels. With this in mind, several natural questions arise - how can we audit these black box systems? And how can we ascertain that the auditee is complying with the audit in good faith? In this work, we rigorously formalize this problem and devise a defense against adversarial attacks on perturbation-based explainers. We propose algorithms for the detection (CAD-Detect) and defense (CAD-Defend) of these attacks, which are aided by our novel conditional anomaly detection approach, KNN-CAD. We demonstrate that our approach successfully detects whether a black box system adversarially conceals its decision-making process and mitigates the adversarial attack on real-world data for the prevalent explainers, LIME and SHAP.



## **33. Robust Graph Representation Learning via Predictive Coding**

cs.LG

27 Pages, 31 Figures

**SubmitDate**: 2022-12-09    [abs](http://arxiv.org/abs/2212.04656v1) [paper-pdf](http://arxiv.org/pdf/2212.04656v1)

**Authors**: Billy Byiringiro, Tommaso Salvatori, Thomas Lukasiewicz

**Abstract**: Predictive coding is a message-passing framework initially developed to model information processing in the brain, and now also topic of research in machine learning due to some interesting properties. One of such properties is the natural ability of generative models to learn robust representations thanks to their peculiar credit assignment rule, that allows neural activities to converge to a solution before updating the synaptic weights. Graph neural networks are also message-passing models, which have recently shown outstanding results in diverse types of tasks in machine learning, providing interdisciplinary state-of-the-art performance on structured data. However, they are vulnerable to imperceptible adversarial attacks, and unfit for out-of-distribution generalization. In this work, we address this by building models that have the same structure of popular graph neural network architectures, but rely on the message-passing rule of predictive coding. Through an extensive set of experiments, we show that the proposed models are (i) comparable to standard ones in terms of performance in both inductive and transductive tasks, (ii) better calibrated, and (iii) robust against multiple kinds of adversarial attacks.



## **34. Effective and Imperceptible Adversarial Textual Attack via Multi-objectivization**

cs.CL

**SubmitDate**: 2022-12-09    [abs](http://arxiv.org/abs/2111.01528v3) [paper-pdf](http://arxiv.org/pdf/2111.01528v3)

**Authors**: Shengcai Liu, Ning Lu, Wenjing Hong, Chao Qian, Ke Tang

**Abstract**: The field of adversarial textual attack has significantly grown over the last few years, where the commonly considered objective is to craft adversarial examples (AEs) that can successfully fool the target model. However, the imperceptibility of attacks, which is also essential for practical attackers, is often left out by previous studies. In consequence, the crafted AEs tend to have obvious structural and semantic differences from the original human-written texts, making them easily perceptible. In this work, we advocate leveraging multi-objectivization to address such issue. Specifically, we formulate the problem of crafting AEs as a multi-objective optimization problem, where the imperceptibility of attacks is considered as auxiliary objectives. Then, we propose a simple yet effective evolutionary algorithm, dubbed HydraText, to solve this problem. To the best of our knowledge, HydraText is currently the only approach that can be effectively applied to both score-based and decision-based attack settings. Exhaustive experiments involving 44237 instances demonstrate that HydraText consistently achieves competitive attack success rates and better attack imperceptibility than the recently proposed attack approaches. A human evaluation study also shows that the AEs crafted by HydraText are more indistinguishable from human-written texts. Finally, these AEs exhibit good transferability and can bring notable robustness improvement to the target model by adversarial training.



## **35. Traditional Classification Neural Networks are Good Generators: They are Competitive with DDPMs and GANs**

cs.CV

This paper has 29 pages with 22 figures, including rich supplementary  information. Project page is at  \url{https://classifier-as-generator.github.io/}

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2211.14794v2) [paper-pdf](http://arxiv.org/pdf/2211.14794v2)

**Authors**: Guangrun Wang, Philip H. S. Torr

**Abstract**: Classifiers and generators have long been separated. We break down this separation and showcase that conventional neural network classifiers can generate high-quality images of a large number of categories, being comparable to the state-of-the-art generative models (e.g., DDPMs and GANs). We achieve this by computing the partial derivative of the classification loss function with respect to the input to optimize the input to produce an image. Since it is widely known that directly optimizing the inputs is similar to targeted adversarial attacks incapable of generating human-meaningful images, we propose a mask-based stochastic reconstruction module to make the gradients semantic-aware to synthesize plausible images. We further propose a progressive-resolution technique to guarantee fidelity, which produces photorealistic images. Furthermore, we introduce a distance metric loss and a non-trivial distribution loss to ensure classification neural networks can synthesize diverse and high-fidelity images. Using traditional neural network classifiers, we can generate good-quality images of 256$\times$256 resolution on ImageNet. Intriguingly, our method is also applicable to text-to-image generation by regarding image-text foundation models as generalized classifiers.   Proving that classifiers have learned the data distribution and are ready for image generation has far-reaching implications, for classifiers are much easier to train than generative models like DDPMs and GANs. We don't even need to train classification models because tons of public ones are available for download. Also, this holds great potential for the interpretability and robustness of classifiers. Project page is at \url{https://classifier-as-generator.github.io/}.



## **36. Universal codes in the shared-randomness model for channels with general distortion capabilities**

cs.IT

Removed the mentioning of online matching, which is not used here

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2007.02330v5) [paper-pdf](http://arxiv.org/pdf/2007.02330v5)

**Authors**: Bruno Bauwens, Marius Zimand

**Abstract**: We put forth new models for universal channel coding. Unlike standard codes which are designed for a specific type of channel, our most general universal code makes communication resilient on every channel, provided the noise level is below the tolerated bound, where the noise level t of a channel is the logarithm of its ambiguity (the maximum number of strings that can be distorted into a given one). The other more restricted universal codes still work for large classes of natural channels. In a universal code, encoding is channel-independent, but the decoding function knows the type of channel. We allow the encoding and the decoding functions to share randomness, which is unavailable to the channel. There are two scenarios for the type of attack that a channel can perform. In the oblivious scenario, codewords belong to an additive group and the channel distorts a codeword by adding a vector from a fixed set. The selection is based on the message and the encoding function, but not on the codeword. In the Hamming scenario, the channel knows the codeword and is fully adversarial. For a universal code, there are two parameters of interest: the rate, which is the ratio between the message length k and the codeword length n, and the number of shared random bits. We show the existence in both scenarios of universal codes with rate 1-t/n - o(1), which is optimal modulo the o(1) term. The number of shared random bits is O(log n) in the oblivious scenario, and O(n) in the Hamming scenario, which, for typical values of the noise level, we show to be optimal, modulo the constant hidden in the O() notation. In both scenarios, the universal encoding is done in time polynomial in n, but the channel-dependent decoding procedures are in general not efficient. For some weaker classes of channels we construct universal codes with polynomial-time encoding and decoding.



## **37. Decorrelative Network Architecture for Robust Electrocardiogram Classification**

cs.LG

16 pages, 6 figures

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2207.09031v2) [paper-pdf](http://arxiv.org/pdf/2207.09031v2)

**Authors**: Christopher Wiedeman, Ge Wang

**Abstract**: Artificial intelligence has made great progress in medical data analysis, but the lack of robustness and trustworthiness has kept these methods from being widely deployed. As it is not possible to train networks that are accurate in all situations, models must recognize situations where they cannot operate confidently. Bayesian deep learning methods sample the model parameter space to estimate uncertainty, but these parameters are often subject to the same vulnerabilities, which can be exploited by adversarial attacks. We propose a novel ensemble approach based on feature decorrelation and Fourier partitioning for teaching networks diverse complementary features, reducing the chance of perturbation-based fooling. We test our approach on electrocardiogram classification, demonstrating superior accuracy confidence measurement, on a variety of adversarial attacks. For example, on our ensemble trained with both decorrelation and Fourier partitioning scored a 50.18% inference accuracy and 48.01% uncertainty accuracy (area under the curve) on {\epsilon} = 50 projected gradient descent attacks, while a conventionally trained ensemble scored 21.1% and 30.31% on these metrics respectively. Our approach does not require expensive optimization with adversarial samples and can be scaled to large problems. These methods can easily be applied to other tasks for more robust and trustworthy models.



## **38. GARNET: Reduced-Rank Topology Learning for Robust and Scalable Graph Neural Networks**

cs.LG

Published as a conference paper at LoG 2022

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2201.12741v5) [paper-pdf](http://arxiv.org/pdf/2201.12741v5)

**Authors**: Chenhui Deng, Xiuyu Li, Zhuo Feng, Zhiru Zhang

**Abstract**: Graph neural networks (GNNs) have been increasingly deployed in various applications that involve learning on non-Euclidean data. However, recent studies show that GNNs are vulnerable to graph adversarial attacks. Although there are several defense methods to improve GNN robustness by eliminating adversarial components, they may also impair the underlying clean graph structure that contributes to GNN training. In addition, few of those defense models can scale to large graphs due to their high computational complexity and memory usage. In this paper, we propose GARNET, a scalable spectral method to boost the adversarial robustness of GNN models. GARNET first leverages weighted spectral embedding to construct a base graph, which is not only resistant to adversarial attacks but also contains critical (clean) graph structure for GNN training. Next, GARNET further refines the base graph by pruning additional uncritical edges based on probabilistic graphical model. GARNET has been evaluated on various datasets, including a large graph with millions of nodes. Our extensive experiment results show that GARNET achieves adversarial accuracy improvement and runtime speedup over state-of-the-art GNN (defense) models by up to 13.27% and 14.7x, respectively.



## **39. RoVISQ: Reduction of Video Service Quality via Adversarial Attacks on Deep Learning-based Video Compression**

cs.CV

Accepted at NDSS 2023

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2203.10183v3) [paper-pdf](http://arxiv.org/pdf/2203.10183v3)

**Authors**: Jung-Woo Chang, Mojan Javaheripi, Seira Hidano, Farinaz Koushanfar

**Abstract**: Video compression plays a crucial role in video streaming and classification systems by maximizing the end-user quality of experience (QoE) at a given bandwidth budget. In this paper, we conduct the first systematic study for adversarial attacks on deep learning-based video compression and downstream classification systems. Our attack framework, dubbed RoVISQ, manipulates the Rate-Distortion ($\textit{R}$-$\textit{D}$) relationship of a video compression model to achieve one or both of the following goals: (1) increasing the network bandwidth, (2) degrading the video quality for end-users. We further devise new objectives for targeted and untargeted attacks to a downstream video classification service. Finally, we design an input-invariant perturbation that universally disrupts video compression and classification systems in real time. Unlike previously proposed attacks on video classification, our adversarial perturbations are the first to withstand compression. We empirically show the resilience of RoVISQ attacks against various defenses, i.e., adversarial training, video denoising, and JPEG compression. Our extensive experimental results on various video datasets show RoVISQ attacks deteriorate peak signal-to-noise ratio by up to 5.6dB and the bit-rate by up to $\sim$ 2.4$\times$ while achieving over 90$\%$ attack success rate on a downstream classifier. Our user study further demonstrates the effect of RoVISQ attacks on users' QoE.



## **40. Contrastive Weighted Learning for Near-Infrared Gaze Estimation**

cs.CV

There were deficiencies in the experiment process for validation

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2211.03073v2) [paper-pdf](http://arxiv.org/pdf/2211.03073v2)

**Authors**: Adam Lee

**Abstract**: Appearance-based gaze estimation has been very successful with the use of deep learning. Many following works improved domain generalization for gaze estimation. However, even though there has been much progress in domain generalization for gaze estimation, most of the recent work have been focused on cross-dataset performance -- accounting for different distributions in illuminations, head pose, and lighting. Although improving gaze estimation in different distributions of RGB images is important, near-infrared image based gaze estimation is also critical for gaze estimation in dark settings. Also there are inherent limitations relying solely on supervised learning for regression tasks. This paper contributes to solving these problems and proposes GazeCWL, a novel framework for gaze estimation with near-infrared images using contrastive learning. This leverages adversarial attack techniques for data augmentation and a novel contrastive loss function specifically for regression tasks that effectively clusters the features of different samples in the latent space. Our model outperforms previous domain generalization models in infrared image based gaze estimation and outperforms the baseline by 45.6\% while improving the state-of-the-art by 8.6\%, we demonstrate the efficacy of our method.



## **41. Targeted Adversarial Attacks against Neural Network Trajectory Predictors**

cs.LG

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2212.04138v1) [paper-pdf](http://arxiv.org/pdf/2212.04138v1)

**Authors**: Kaiyuan Tan, Jun Wang, Yiannis Kantaros

**Abstract**: Trajectory prediction is an integral component of modern autonomous systems as it allows for envisioning future intentions of nearby moving agents. Due to the lack of other agents' dynamics and control policies, deep neural network (DNN) models are often employed for trajectory forecasting tasks. Although there exists an extensive literature on improving the accuracy of these models, there is a very limited number of works studying their robustness against adversarially crafted input trajectories. To bridge this gap, in this paper, we propose a targeted adversarial attack against DNN models for trajectory forecasting tasks. We call the proposed attack TA4TP for Targeted adversarial Attack for Trajectory Prediction. Our approach generates adversarial input trajectories that are capable of fooling DNN models into predicting user-specified target/desired trajectories. Our attack relies on solving a nonlinear constrained optimization problem where the objective function captures the deviation of the predicted trajectory from a target one while the constraints model physical requirements that the adversarial input should satisfy. The latter ensures that the inputs look natural and they are safe to execute (e.g., they are close to nominal inputs and away from obstacles). We demonstrate the effectiveness of TA4TP on two state-of-the-art DNN models and two datasets. To the best of our knowledge, we propose the first targeted adversarial attack against DNN models used for trajectory forecasting.



## **42. ADMM based Distributed State Observer Design under Sparse Sensor Attacks**

eess.SY

7 pages, 6 figures

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2209.06292v2) [paper-pdf](http://arxiv.org/pdf/2209.06292v2)

**Authors**: Vinaya Mary Prinse, Rachel Kalpana Kalaimani

**Abstract**: This paper considers the design of a distributed state-observer for discrete-time Linear Time-Invariant (LTI) systems in the presence of sensor attacks. We assume there is a network of observer nodes, communicating with each other over an undirected graph, each with partial measurements of the output corrupted by some adversarial attack. We address the case of sparse attacks where the attacker targets a small subset of sensors. An algorithm based on Alternating Direction Method of Multipliers (ADMM) is developed which provides an update law for each observer which ensures convergence of each observer node to the actual state asymptotically.



## **43. Task and Model Agnostic Adversarial Attack on Graph Neural Networks**

cs.LG

To appear as a full paper in AAAI 2023

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2112.13267v3) [paper-pdf](http://arxiv.org/pdf/2112.13267v3)

**Authors**: Kartik Sharma, Samidha Verma, Sourav Medya, Arnab Bhattacharya, Sayan Ranu

**Abstract**: Adversarial attacks on Graph Neural Networks (GNNs) reveal their security vulnerabilities, limiting their adoption in safety-critical applications. However, existing attack strategies rely on the knowledge of either the GNN model being used or the predictive task being attacked. Is this knowledge necessary? For example, a graph may be used for multiple downstream tasks unknown to a practical attacker. It is thus important to test the vulnerability of GNNs to adversarial perturbations in a model and task agnostic setting. In this work, we study this problem and show that GNNs remain vulnerable even when the downstream task and model are unknown. The proposed algorithm, TANDIS (Targeted Attack via Neighborhood DIStortion) shows that distortion of node neighborhoods is effective in drastically compromising prediction performance. Although neighborhood distortion is an NP-hard problem, TANDIS designs an effective heuristic through a novel combination of Graph Isomorphism Network with deep Q-learning. Extensive experiments on real datasets and state-of-the-art models show that, on average, TANDIS is up to 50% more effective than state-of-the-art techniques, while being more than 1000 times faster.



## **44. SwitchX: Gmin-Gmax Switching for Energy-Efficient and Robust Implementation of Binary Neural Networks on ReRAM Xbars**

cs.ET

Accepted to ACM Transactions on Design Automation of Electronic  Systems on 30 Nov 2022

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2011.14498v3) [paper-pdf](http://arxiv.org/pdf/2011.14498v3)

**Authors**: Abhiroop Bhattacharjee, Priyadarshini Panda

**Abstract**: Memristive crossbars can efficiently implement Binarized Neural Networks (BNNs) wherein the weights are stored in high-resistance states (HRS) and low-resistance states (LRS) of the synapses. We propose SwitchX mapping of BNN weights onto ReRAM crossbars such that the impact of crossbar non-idealities, that lead to degradation in computational accuracy, are minimized. Essentially, SwitchX maps the binary weights in such manner that a crossbar instance comprises of more HRS than LRS synapses. We find BNNs mapped onto crossbars with SwitchX to exhibit better robustness against adversarial attacks than the standard crossbar-mapped BNNs, the baseline. Finally, we combine SwitchX with state-aware training (that further increases the feasibility of HRS states during weight mapping) to boost the robustness of a BNN on hardware. We find that this approach yields stronger defense against adversarial attacks than adversarial training, a state-of-the-art software defense. We perform experiments on a VGG16 BNN with benchmark datasets (CIFAR-10, CIFAR-100 & TinyImagenet) and use Fast Gradient Sign Method and Projected Gradient Descent adversarial attacks. We show that SwitchX combined with state-aware training can yield upto ~35% improvements in clean accuracy and ~6-16% in adversarial accuracies against conventional BNNs. Furthermore, an important by-product of SwitchX mapping is increased crossbar power savings, owing to an increased proportion of HRS synapses, that is furthered with state-aware training. We obtain upto ~21-22% savings in crossbar power consumption for state-aware trained BNN mapped via SwitchX on 16x16 & 32x32 crossbars using the CIFAR-10 & CIFAR-100 datasets.



## **45. Learning Polysemantic Spoof Trace: A Multi-Modal Disentanglement Network for Face Anti-spoofing**

cs.CV

Accepted by AAAI 2023

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2212.03943v1) [paper-pdf](http://arxiv.org/pdf/2212.03943v1)

**Authors**: Kaicheng Li, Hongyu Yang, Binghui Chen, Pengyu Li, Biao Wang, Di Huang

**Abstract**: Along with the widespread use of face recognition systems, their vulnerability has become highlighted. While existing face anti-spoofing methods can be generalized between attack types, generic solutions are still challenging due to the diversity of spoof characteristics. Recently, the spoof trace disentanglement framework has shown great potential for coping with both seen and unseen spoof scenarios, but the performance is largely restricted by the single-modal input. This paper focuses on this issue and presents a multi-modal disentanglement model which targetedly learns polysemantic spoof traces for more accurate and robust generic attack detection. In particular, based on the adversarial learning mechanism, a two-stream disentangling network is designed to estimate spoof patterns from the RGB and depth inputs, respectively. In this case, it captures complementary spoofing clues inhering in different attacks. Furthermore, a fusion module is exploited, which recalibrates both representations at multiple stages to promote the disentanglement in each individual modality. It then performs cross-modality aggregation to deliver a more comprehensive spoof trace representation for prediction. Extensive evaluations are conducted on multiple benchmarks, demonstrating that learning polysemantic spoof traces favorably contributes to anti-spoofing with more perceptible and interpretable results.



## **46. Multiple Perturbation Attack: Attack Pixelwise Under Different $\ell_p$-norms For Better Adversarial Performance**

cs.CV

18 pages, 8 figures, 7 tables

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2212.03069v2) [paper-pdf](http://arxiv.org/pdf/2212.03069v2)

**Authors**: Ngoc N. Tran, Anh Tuan Bui, Dinh Phung, Trung Le

**Abstract**: Adversarial machine learning has been both a major concern and a hot topic recently, especially with the ubiquitous use of deep neural networks in the current landscape. Adversarial attacks and defenses are usually likened to a cat-and-mouse game in which defenders and attackers evolve over the time. On one hand, the goal is to develop strong and robust deep networks that are resistant to malicious actors. On the other hand, in order to achieve that, we need to devise even stronger adversarial attacks to challenge these defense models. Most of existing attacks employs a single $\ell_p$ distance (commonly, $p\in\{1,2,\infty\}$) to define the concept of closeness and performs steepest gradient ascent w.r.t. this $p$-norm to update all pixels in an adversarial example in the same way. These $\ell_p$ attacks each has its own pros and cons; and there is no single attack that can successfully break through defense models that are robust against multiple $\ell_p$ norms simultaneously. Motivated by these observations, we come up with a natural approach: combining various $\ell_p$ gradient projections on a pixel level to achieve a joint adversarial perturbation. Specifically, we learn how to perturb each pixel to maximize the attack performance, while maintaining the overall visual imperceptibility of adversarial examples. Finally, through various experiments with standardized benchmarks, we show that our method outperforms most current strong attacks across state-of-the-art defense mechanisms, while retaining its ability to remain clean visually.



## **47. Universal Backdoor Attacks Detection via Adaptive Adversarial Probe**

cs.CV

8 pages, 8 figures

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2209.05244v3) [paper-pdf](http://arxiv.org/pdf/2209.05244v3)

**Authors**: Yuhang Wang, Huafeng Shi, Rui Min, Ruijia Wu, Siyuan Liang, Yichao Wu, Ding Liang, Aishan Liu

**Abstract**: Extensive evidence has demonstrated that deep neural networks (DNNs) are vulnerable to backdoor attacks, which motivates the development of backdoor attacks detection. Most detection methods are designed to verify whether a model is infected with presumed types of backdoor attacks, yet the adversary is likely to generate diverse backdoor attacks in practice that are unforeseen to defenders, which challenge current detection strategies. In this paper, we focus on this more challenging scenario and propose a universal backdoor attacks detection method named Adaptive Adversarial Probe (A2P). Specifically, we posit that the challenge of universal backdoor attacks detection lies in the fact that different backdoor attacks often exhibit diverse characteristics in trigger patterns (i.e., sizes and transparencies). Therefore, our A2P adopts a global-to-local probing framework, which adversarially probes images with adaptive regions/budgets to fit various backdoor triggers of different sizes/transparencies. Regarding the probing region, we propose the attention-guided region generation strategy that generates region proposals with different sizes/locations based on the attention of the target model, since trigger regions often manifest higher model activation. Considering the attack budget, we introduce the box-to-sparsity scheduling that iteratively increases the perturbation budget from box to sparse constraint, so that we could better activate different latent backdoors with different transparencies. Extensive experiments on multiple datasets (CIFAR-10, GTSRB, Tiny-ImageNet) demonstrate that our method outperforms state-of-the-art baselines by large margins (+12%).



## **48. LeNo: Adversarial Robust Salient Object Detection Networks with Learnable Noise**

cs.CV

8 pages, 6 figures, accepted by AAAI 2023

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2210.15392v2) [paper-pdf](http://arxiv.org/pdf/2210.15392v2)

**Authors**: He Wang, Lin Wan, He Tang

**Abstract**: Pixel-wise prediction with deep neural network has become an effective paradigm for salient object detection (SOD) and achieved remarkable performance. However, very few SOD models are robust against adversarial attacks which are visually imperceptible for human visual attention. The previous work robust saliency (ROSA) shuffles the pre-segmented superpixels and then refines the coarse saliency map by the densely connected conditional random field (CRF). Different from ROSA that relies on various pre- and post-processings, this paper proposes a light-weight Learnable Noise (LeNo) to defend adversarial attacks for SOD models. LeNo preserves accuracy of SOD models on both adversarial and clean images, as well as inference speed. In general, LeNo consists of a simple shallow noise and noise estimation that embedded in the encoder and decoder of arbitrary SOD networks respectively. Inspired by the center prior of human visual attention mechanism, we initialize the shallow noise with a cross-shaped gaussian distribution for better defense against adversarial attacks. Instead of adding additional network components for post-processing, the proposed noise estimation modifies only one channel of the decoder. With the deeply-supervised noise-decoupled training on state-of-the-art RGB and RGB-D SOD networks, LeNo outperforms previous works not only on adversarial images but also on clean images, which contributes stronger robustness for SOD. Our code is available at https://github.com/ssecv/LeNo.



## **49. Pre-trained Encoders in Self-Supervised Learning Improve Secure and Privacy-preserving Supervised Learning**

cs.CR

**SubmitDate**: 2022-12-06    [abs](http://arxiv.org/abs/2212.03334v1) [paper-pdf](http://arxiv.org/pdf/2212.03334v1)

**Authors**: Hongbin Liu, Wenjie Qu, Jinyuan Jia, Neil Zhenqiang Gong

**Abstract**: Classifiers in supervised learning have various security and privacy issues, e.g., 1) data poisoning attacks, backdoor attacks, and adversarial examples on the security side as well as 2) inference attacks and the right to be forgotten for the training data on the privacy side. Various secure and privacy-preserving supervised learning algorithms with formal guarantees have been proposed to address these issues. However, they suffer from various limitations such as accuracy loss, small certified security guarantees, and/or inefficiency. Self-supervised learning is an emerging technique to pre-train encoders using unlabeled data. Given a pre-trained encoder as a feature extractor, supervised learning can train a simple yet accurate classifier using a small amount of labeled training data. In this work, we perform the first systematic, principled measurement study to understand whether and when a pre-trained encoder can address the limitations of secure or privacy-preserving supervised learning algorithms. Our key findings are that a pre-trained encoder substantially improves 1) both accuracy under no attacks and certified security guarantees against data poisoning and backdoor attacks of state-of-the-art secure learning algorithms (i.e., bagging and KNN), 2) certified security guarantees of randomized smoothing against adversarial examples without sacrificing its accuracy under no attacks, 3) accuracy of differentially private classifiers, and 4) accuracy and/or efficiency of exact machine unlearning.



## **50. Robust Models are less Over-Confident**

cs.CV

accepted at NeurIPS 2022

**SubmitDate**: 2022-12-06    [abs](http://arxiv.org/abs/2210.05938v2) [paper-pdf](http://arxiv.org/pdf/2210.05938v2)

**Authors**: Julia Grabinski, Paul Gavrikov, Janis Keuper, Margret Keuper

**Abstract**: Despite the success of convolutional neural networks (CNNs) in many academic benchmarks for computer vision tasks, their application in the real-world is still facing fundamental challenges. One of these open problems is the inherent lack of robustness, unveiled by the striking effectiveness of adversarial attacks. Current attack methods are able to manipulate the network's prediction by adding specific but small amounts of noise to the input. In turn, adversarial training (AT) aims to achieve robustness against such attacks and ideally a better model generalization ability by including adversarial samples in the trainingset. However, an in-depth analysis of the resulting robust models beyond adversarial robustness is still pending. In this paper, we empirically analyze a variety of adversarially trained models that achieve high robust accuracies when facing state-of-the-art attacks and we show that AT has an interesting side-effect: it leads to models that are significantly less overconfident with their decisions, even on clean data than non-robust models. Further, our analysis of robust models shows that not only AT but also the model's building blocks (like activation functions and pooling) have a strong influence on the models' prediction confidences. Data & Project website: https://github.com/GeJulia/robustness_confidences_evaluation



