# Latest Adversarial Attack Papers
**update at 2022-07-28 06:31:24**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Making Corgis Important for Honeycomb Classification: Adversarial Attacks on Concept-based Explainability Tools**

cs.LG

AdvML Frontiers 2022 @ ICML 2022 workshop

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2110.07120v2)

**Authors**: Davis Brown, Henry Kvinge

**Abstracts**: Methods for model explainability have become increasingly critical for testing the fairness and soundness of deep learning. Concept-based interpretability techniques, which use a small set of human-interpretable concept exemplars in order to measure the influence of a concept on a model's internal representation of input, are an important thread in this line of research. In this work we show that these explainability methods can suffer the same vulnerability to adversarial attacks as the models they are meant to analyze. We demonstrate this phenomenon on two well-known concept-based interpretability methods: TCAV and faceted feature visualization. We show that by carefully perturbing the examples of the concept that is being investigated, we can radically change the output of the interpretability method. The attacks that we propose can either induce positive interpretations (polka dots are an important concept for a model when classifying zebras) or negative interpretations (stripes are not an important factor in identifying images of a zebra). Our work highlights the fact that in safety-critical applications, there is need for security around not only the machine learning pipeline but also the model interpretation process.



## **2. TnT Attacks! Universal Naturalistic Adversarial Patches Against Deep Neural Network Systems**

cs.CV

Accepted for publication in the IEEE Transactions on Information  Forensics & Security (TIFS)

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2111.09999v2)

**Authors**: Bao Gia Doan, Minhui Xue, Shiqing Ma, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstracts**: Deep neural networks are vulnerable to attacks from adversarial inputs and, more recently, Trojans to misguide or hijack the model's decision. We expose the existence of an intriguing class of spatially bounded, physically realizable, adversarial examples -- Universal NaTuralistic adversarial paTches -- we call TnTs, by exploring the superset of the spatially bounded adversarial example space and the natural input space within generative adversarial networks. Now, an adversary can arm themselves with a patch that is naturalistic, less malicious-looking, physically realizable, highly effective achieving high attack success rates, and universal. A TnT is universal because any input image captured with a TnT in the scene will: i) misguide a network (untargeted attack); or ii) force the network to make a malicious decision (targeted attack). Interestingly, now, an adversarial patch attacker has the potential to exert a greater level of control -- the ability to choose a location-independent, natural-looking patch as a trigger in contrast to being constrained to noisy perturbations -- an ability is thus far shown to be only possible with Trojan attack methods needing to interfere with the model building processes to embed a backdoor at the risk discovery; but, still realize a patch deployable in the physical world. Through extensive experiments on the large-scale visual classification task, ImageNet with evaluations across its entire validation set of 50,000 images, we demonstrate the realistic threat from TnTs and the robustness of the attack. We show a generalization of the attack to create patches achieving higher attack success rates than existing state-of-the-art methods. Our results show the generalizability of the attack to different visual classification tasks (CIFAR-10, GTSRB, PubFig) and multiple state-of-the-art deep neural networks such as WideResnet50, Inception-V3 and VGG-16.



## **3. Verification-Aided Deep Ensemble Selection**

cs.LG

To appear in FMCAD 2022

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2202.03898v2)

**Authors**: Guy Amir, Tom Zelazny, Guy Katz, Michael Schapira

**Abstracts**: Deep neural networks (DNNs) have become the technology of choice for realizing a variety of complex tasks. However, as highlighted by many recent studies, even an imperceptible perturbation to a correctly classified input can lead to misclassification by a DNN. This renders DNNs vulnerable to strategic input manipulations by attackers, and also oversensitive to environmental noise.   To mitigate this phenomenon, practitioners apply joint classification by an *ensemble* of DNNs. By aggregating the classification outputs of different individual DNNs for the same input, ensemble-based classification reduces the risk of misclassifications due to the specific realization of the stochastic training process of any single DNN. However, the effectiveness of a DNN ensemble is highly dependent on its members *not simultaneously erring* on many different inputs.   In this case study, we harness recent advances in DNN verification to devise a methodology for identifying ensemble compositions that are less prone to simultaneous errors, even when the input is adversarially perturbed -- resulting in more robustly-accurate ensemble-based classification.   Our proposed framework uses a DNN verifier as a backend, and includes heuristics that help reduce the high complexity of directly verifying ensembles. More broadly, our work puts forth a novel universal objective for formal verification that can potentially improve the robustness of real-world, deep-learning-based systems across a variety of application domains.



## **4. $p$-DkNN: Out-of-Distribution Detection Through Statistical Testing of Deep Representations**

cs.LG

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12545v1)

**Authors**: Adam Dziedzic, Stephan Rabanser, Mohammad Yaghini, Armin Ale, Murat A. Erdogdu, Nicolas Papernot

**Abstracts**: The lack of well-calibrated confidence estimates makes neural networks inadequate in safety-critical domains such as autonomous driving or healthcare. In these settings, having the ability to abstain from making a prediction on out-of-distribution (OOD) data can be as important as correctly classifying in-distribution data. We introduce $p$-DkNN, a novel inference procedure that takes a trained deep neural network and analyzes the similarity structures of its intermediate hidden representations to compute $p$-values associated with the end-to-end model prediction. The intuition is that statistical tests performed on latent representations can serve not only as a classifier, but also offer a statistically well-founded estimation of uncertainty. $p$-DkNN is scalable and leverages the composition of representations learned by hidden layers, which makes deep representation learning successful. Our theoretical analysis builds on Neyman-Pearson classification and connects it to recent advances in selective classification (reject option). We demonstrate advantageous trade-offs between abstaining from predicting on OOD inputs and maintaining high accuracy on in-distribution inputs. We find that $p$-DkNN forces adaptive attackers crafting adversarial examples, a form of worst-case OOD inputs, to introduce semantically meaningful changes to the inputs.



## **5. TAFIM: Targeted Adversarial Attacks against Facial Image Manipulations**

cs.CV

(ECCV 2022 Paper) Video: https://youtu.be/11VMOJI7tKg Project Page:  https://shivangi-aneja.github.io/projects/tafim/

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2112.09151v2)

**Authors**: Shivangi Aneja, Lev Markhasin, Matthias Niessner

**Abstracts**: Face manipulation methods can be misused to affect an individual's privacy or to spread disinformation. To this end, we introduce a novel data-driven approach that produces image-specific perturbations which are embedded in the original images. The key idea is that these protected images prevent face manipulation by causing the manipulation model to produce a predefined manipulation target (uniformly colored output image in our case) instead of the actual manipulation. In addition, we propose to leverage differentiable compression approximation, hence making generated perturbations robust to common image compression. In order to prevent against multiple manipulation methods simultaneously, we further propose a novel attention-based fusion of manipulation-specific perturbations. Compared to traditional adversarial attacks that optimize noise patterns for each image individually, our generalized model only needs a single forward pass, thus running orders of magnitude faster and allowing for easy integration in image processing stacks, even on resource-constrained devices like smartphones.



## **6. SegPGD: An Effective and Efficient Adversarial Attack for Evaluating and Boosting Segmentation Robustness**

cs.CV

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12391v1)

**Authors**: Jindong Gu, Hengshuang Zhao, Volker Tresp, Philip Torr

**Abstracts**: Deep neural network-based image classifications are vulnerable to adversarial perturbations. The image classifications can be easily fooled by adding artificial small and imperceptible perturbations to input images. As one of the most effective defense strategies, adversarial training was proposed to address the vulnerability of classification models, where the adversarial examples are created and injected into training data during training. The attack and defense of classification models have been intensively studied in past years. Semantic segmentation, as an extension of classifications, has also received great attention recently. Recent work shows a large number of attack iterations are required to create effective adversarial examples to fool segmentation models. The observation makes both robustness evaluation and adversarial training on segmentation models challenging. In this work, we propose an effective and efficient segmentation attack method, dubbed SegPGD. Besides, we provide a convergence analysis to show the proposed SegPGD can create more effective adversarial examples than PGD under the same number of attack iterations. Furthermore, we propose to apply our SegPGD as the underlying attack method for segmentation adversarial training. Since SegPGD can create more effective adversarial examples, the adversarial training with our SegPGD can boost the robustness of segmentation models. Our proposals are also verified with experiments on popular Segmentation model architectures and standard segmentation datasets.



## **7. Adversarial Attack across Datasets**

cs.CV

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2110.07718v2)

**Authors**: Yunxiao Qin, Yuanhao Xiong, Jinfeng Yi, Lihong Cao, Cho-Jui Hsieh

**Abstracts**: Existing transfer attack methods commonly assume that the attacker knows the training set (e.g., the label set, the input size) of the black-box victim models, which is usually unrealistic because in some cases the attacker cannot know this information. In this paper, we define a Generalized Transferable Attack (GTA) problem where the attacker doesn't know this information and is acquired to attack any randomly encountered images that may come from unknown datasets. To solve the GTA problem, we propose a novel Image Classification Eraser (ICE) that trains a particular attacker to erase classification information of any images from arbitrary datasets. Experiments on several datasets demonstrate that ICE greatly outperforms existing transfer attacks on GTA, and show that ICE uses similar texture-like noises to perturb different images from different datasets. Moreover, fast fourier transformation analysis indicates that the main components in each ICE noise are three sine waves for the R, G, and B image channels. Inspired by this interesting finding, we then design a novel Sine Attack (SA) method to optimize the three sine waves. Experiments show that SA performs comparably to ICE, indicating that the three sine waves are effective and enough to break DNNs under the GTA setting.



## **8. Improving Adversarial Robustness via Mutual Information Estimation**

cs.LG

This version has modified Eq.2 and its proof in the published version

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12203v1)

**Authors**: Dawei Zhou, Nannan Wang, Xinbo Gao, Bo Han, Xiaoyu Wang, Yibing Zhan, Tongliang Liu

**Abstracts**: Deep neural networks (DNNs) are found to be vulnerable to adversarial noise. They are typically misled by adversarial samples to make wrong predictions. To alleviate this negative effect, in this paper, we investigate the dependence between outputs of the target model and input adversarial samples from the perspective of information theory, and propose an adversarial defense method. Specifically, we first measure the dependence by estimating the mutual information (MI) between outputs and the natural patterns of inputs (called natural MI) and MI between outputs and the adversarial patterns of inputs (called adversarial MI), respectively. We find that adversarial samples usually have larger adversarial MI and smaller natural MI compared with those w.r.t. natural samples. Motivated by this observation, we propose to enhance the adversarial robustness by maximizing the natural MI and minimizing the adversarial MI during the training process. In this way, the target model is expected to pay more attention to the natural pattern that contains objective semantics. Empirical evaluations demonstrate that our method could effectively improve the adversarial accuracy against multiple attacks.



## **9. Versatile Weight Attack via Flipping Limited Bits**

cs.CR

Extension of our ICLR 2021 work: arXiv:2102.10496

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12405v1)

**Authors**: Jiawang Bai, Baoyuan Wu, Zhifeng Li, Shu-tao Xia

**Abstracts**: To explore the vulnerability of deep neural networks (DNNs), many attack paradigms have been well studied, such as the poisoning-based backdoor attack in the training stage and the adversarial attack in the inference stage. In this paper, we study a novel attack paradigm, which modifies model parameters in the deployment stage. Considering the effectiveness and stealthiness goals, we provide a general formulation to perform the bit-flip based weight attack, where the effectiveness term could be customized depending on the attacker's purpose. Furthermore, we present two cases of the general formulation with different malicious purposes, i.e., single sample attack (SSA) and triggered samples attack (TSA). To this end, we formulate this problem as a mixed integer programming (MIP) to jointly determine the state of the binary bits (0 or 1) in the memory and learn the sample modification. Utilizing the latest technique in integer programming, we equivalently reformulate this MIP problem as a continuous optimization problem, which can be effectively and efficiently solved using the alternating direction method of multipliers (ADMM) method. Consequently, the flipped critical bits can be easily determined through optimization, rather than using a heuristic strategy. Extensive experiments demonstrate the superiority of SSA and TSA in attacking DNNs.



## **10. Privacy Against Inference Attacks in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2022-07-24    [paper-pdf](http://arxiv.org/pdf/2207.11788v1)

**Authors**: Borzoo Rassouli, Morteza Varasteh, Deniz Gunduz

**Abstracts**: Vertical federated learning is considered, where an active party, having access to true class labels, wishes to build a classification model by utilizing more features from a passive party, which has no access to the labels, to improve the model accuracy. In the prediction phase, with logistic regression as the classification model, several inference attack techniques are proposed that the adversary, i.e., the active party, can employ to reconstruct the passive party's features, regarded as sensitive information. These attacks, which are mainly based on a classical notion of the center of a set, i.e., the Chebyshev center, are shown to be superior to those proposed in the literature. Moreover, several theoretical performance guarantees are provided for the aforementioned attacks. Subsequently, we consider the minimum amount of information that the adversary needs to fully reconstruct the passive party's features. In particular, it is shown that when the passive party holds one feature, and the adversary is only aware of the signs of the parameters involved, it can perfectly reconstruct that feature when the number of predictions is large enough. Next, as a defense mechanism, two privacy-preserving schemes are proposed that worsen the adversary's reconstruction attacks, while preserving the full benefits that VFL brings to the active party. Finally, experimental results demonstrate the effectiveness of the proposed attacks and the privacy-preserving schemes.



## **11. Can we achieve robustness from data alone?**

cs.LG

**SubmitDate**: 2022-07-24    [paper-pdf](http://arxiv.org/pdf/2207.11727v1)

**Authors**: Nikolaos Tsilivis, Jingtong Su, Julia Kempe

**Abstracts**: Adversarial training and its variants have come to be the prevailing methods to achieve adversarially robust classification using neural networks. However, its increased computational cost together with the significant gap between standard and robust performance hinder progress and beg the question of whether we can do better. In this work, we take a step back and ask: Can models achieve robustness via standard training on a suitably optimized set? To this end, we devise a meta-learning method for robust classification, that optimizes the dataset prior to its deployment in a principled way, and aims to effectively remove the non-robust parts of the data. We cast our optimization method as a multi-step PGD procedure on kernel regression, with a class of kernels that describe infinitely wide neural nets (Neural Tangent Kernels - NTKs). Experiments on MNIST and CIFAR-10 demonstrate that the datasets we produce enjoy very high robustness against PGD attacks, when deployed in both kernel regression classifiers and neural networks. However, this robustness is somewhat fallacious, as alternative attacks manage to fool the models, which we find to be the case for previous similar works in the literature as well. We discuss potential reasons for this and outline further avenues of research.



## **12. Proving Common Mechanisms Shared by Twelve Methods of Boosting Adversarial Transferability**

cs.LG

**SubmitDate**: 2022-07-24    [paper-pdf](http://arxiv.org/pdf/2207.11694v1)

**Authors**: Quanshi Zhang, Xin Wang, Jie Ren, Xu Cheng, Shuyun Lin, Yisen Wang, Xiangming Zhu

**Abstracts**: Although many methods have been proposed to enhance the transferability of adversarial perturbations, these methods are designed in a heuristic manner, and the essential mechanism for improving adversarial transferability is still unclear. This paper summarizes the common mechanism shared by twelve previous transferability-boosting methods in a unified view, i.e., these methods all reduce game-theoretic interactions between regional adversarial perturbations. To this end, we focus on the attacking utility of all interactions between regional adversarial perturbations, and we first discover and prove the negative correlation between the adversarial transferability and the attacking utility of interactions. Based on this discovery, we theoretically prove and empirically verify that twelve previous transferability-boosting methods all reduce interactions between regional adversarial perturbations. More crucially, we consider the reduction of interactions as the essential reason for the enhancement of adversarial transferability. Furthermore, we design the interaction loss to directly penalize interactions between regional adversarial perturbations during attacking. Experimental results show that the interaction loss significantly improves the transferability of adversarial perturbations.



## **13. Testing the Robustness of Learned Index Structures**

cs.DB

**SubmitDate**: 2022-07-23    [paper-pdf](http://arxiv.org/pdf/2207.11575v1)

**Authors**: Matthias Bachfischer, Renata Borovica-Gajic, Benjamin I. P. Rubinstein

**Abstracts**: While early empirical evidence has supported the case for learned index structures as having favourable average-case performance, little is known about their worst-case performance. By contrast, classical structures are known to achieve optimal worst-case behaviour. This work evaluates the robustness of learned index structures in the presence of adversarial workloads. To simulate adversarial workloads, we carry out a data poisoning attack on linear regression models that manipulates the cumulative distribution function (CDF) on which the learned index model is trained. The attack deteriorates the fit of the underlying ML model by injecting a set of poisoning keys into the training dataset, which leads to an increase in the prediction error of the model and thus deteriorates the overall performance of the learned index structure. We assess the performance of various regression methods and the learned index implementations ALEX and PGM-Index. We show that learned index structures can suffer from a significant performance deterioration of up to 20% when evaluated on poisoned vs. non-poisoned datasets.



## **14. How does Heterophily Impact the Robustness of Graph Neural Networks? Theoretical Connections and Practical Implications**

cs.LG

KDD 2022 camera ready version + full appendix; 20 pages, 2 figures

**SubmitDate**: 2022-07-23    [paper-pdf](http://arxiv.org/pdf/2106.07767v4)

**Authors**: Jiong Zhu, Junchen Jin, Donald Loveland, Michael T. Schaub, Danai Koutra

**Abstracts**: We bridge two research directions on graph neural networks (GNNs), by formalizing the relation between heterophily of node labels (i.e., connected nodes tend to have dissimilar labels) and the robustness of GNNs to adversarial attacks. Our theoretical and empirical analyses show that for homophilous graph data, impactful structural attacks always lead to reduced homophily, while for heterophilous graph data the change in the homophily level depends on the node degrees. These insights have practical implications for defending against attacks on real-world graphs: we deduce that separate aggregators for ego- and neighbor-embeddings, a design principle which has been identified to significantly improve prediction for heterophilous graph data, can also offer increased robustness to GNNs. Our comprehensive experiments show that GNNs merely adopting this design achieve improved empirical and certifiable robustness compared to the best-performing unvaccinated model. Additionally, combining this design with explicit defense mechanisms against adversarial attacks leads to an improved robustness with up to 18.33% performance increase under attacks compared to the best-performing vaccinated model.



## **15. Do Perceptually Aligned Gradients Imply Adversarial Robustness?**

cs.CV

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2207.11378v1)

**Authors**: Roy Ganz, Bahjat Kawar, Michael Elad

**Abstracts**: In the past decade, deep learning-based networks have achieved unprecedented success in numerous tasks, including image classification. Despite this remarkable achievement, recent studies have demonstrated that such networks are easily fooled by small malicious perturbations, also known as adversarial examples. This security weakness led to extensive research aimed at obtaining robust models. Beyond the clear robustness benefits of such models, it was also observed that their gradients with respect to the input align with human perception. Several works have identified Perceptually Aligned Gradients (PAG) as a byproduct of robust training, but none have considered it as a standalone phenomenon nor studied its own implications. In this work, we focus on this trait and test whether Perceptually Aligned Gradients imply Robustness. To this end, we develop a novel objective to directly promote PAG in training classifiers and examine whether models with such gradients are more robust to adversarial attacks. Extensive experiments on CIFAR-10 and STL validate that such models have improved robust performance, exposing the surprising bidirectional connection between PAG and robustness.



## **16. Practical Privacy Attacks on Vertical Federated Learning**

cs.CR

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2011.09290v3)

**Authors**: Haiqin Weng, Juntao Zhang, Xingjun Ma, Feng Xue, Tao Wei, Shouling Ji, Zhiyuan Zong

**Abstracts**: Federated learning (FL) is a privacy-preserving learning paradigm that allows multiple parities to jointly train a powerful machine learning model without sharing their private data. According to the form of collaboration, FL can be further divided into horizontal federated learning (HFL) and vertical federated learning (VFL). In HFL, participants share the same feature space and collaborate on data samples, while in VFL, participants share the same sample IDs and collaborate on features. VFL has a broader scope of applications and is arguably more suitable for joint model training between large enterprises.   In this paper, we focus on VFL and investigate potential privacy leakage in real-world VFL frameworks. We design and implement two practical privacy attacks: reverse multiplication attack for the logistic regression VFL protocol; and reverse sum attack for the XGBoost VFL protocol. We empirically show that the two attacks are (1) effective - the adversary can successfully steal the private training data, even when the intermediate outputs are encrypted to protect data privacy; (2) evasive - the attacks do not deviate from the protocol specification nor deteriorate the accuracy of the target model; and (3) easy - the adversary needs little prior knowledge about the data distribution of the target participant. We also show the leaked information is as effective as the raw training data in training an alternative classifier. We further discuss potential countermeasures and their challenges, which we hope can lead to several promising research directions.



## **17. On Higher Adversarial Susceptibility of Contrastive Self-Supervised Learning**

cs.CV

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2207.10862v1)

**Authors**: Rohit Gupta, Naveed Akhtar, Ajmal Mian, Mubarak Shah

**Abstracts**: Contrastive self-supervised learning (CSL) has managed to match or surpass the performance of supervised learning in image and video classification. However, it is still largely unknown if the nature of the representation induced by the two learning paradigms is similar. We investigate this under the lens of adversarial robustness. Our analytical treatment of the problem reveals intrinsic higher sensitivity of CSL over supervised learning. It identifies the uniform distribution of data representation over a unit hypersphere in the CSL representation space as the key contributor to this phenomenon. We establish that this increases model sensitivity to input perturbations in the presence of false negatives in the training data. Our finding is supported by extensive experiments for image and video classification using adversarial perturbations and other input corruptions. Building on the insights, we devise strategies that are simple, yet effective in improving model robustness with CSL training. We demonstrate up to 68% reduction in the performance gap between adversarially attacked CSL and its supervised counterpart. Finally, we contribute to robust CSL paradigm by incorporating our findings in adversarial self-supervised learning. We demonstrate an average gain of about 5% over two different state-of-the-art methods in this domain.



## **18. Adversarially-Aware Robust Object Detector**

cs.CV

ECCV2022 oral paper

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2207.06202v3)

**Authors**: Ziyi Dong, Pengxu Wei, Liang Lin

**Abstracts**: Object detection, as a fundamental computer vision task, has achieved a remarkable progress with the emergence of deep neural networks. Nevertheless, few works explore the adversarial robustness of object detectors to resist adversarial attacks for practical applications in various real-world scenarios. Detectors have been greatly challenged by unnoticeable perturbation, with sharp performance drop on clean images and extremely poor performance on adversarial images. In this work, we empirically explore the model training for adversarial robustness in object detection, which greatly attributes to the conflict between learning clean images and adversarial images. To mitigate this issue, we propose a Robust Detector (RobustDet) based on adversarially-aware convolution to disentangle gradients for model learning on clean and adversarial images. RobustDet also employs the Adversarial Image Discriminator (AID) and Consistent Features with Reconstruction (CFR) to ensure a reliable robustness. Extensive experiments on PASCAL VOC and MS-COCO demonstrate that our model effectively disentangles gradients and significantly enhances the detection robustness with maintaining the detection ability on clean images.



## **19. Boosting Transferability of Targeted Adversarial Examples via Hierarchical Generative Networks**

cs.LG

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2107.01809v2)

**Authors**: Xiao Yang, Yinpeng Dong, Tianyu Pang, Hang Su, Jun Zhu

**Abstracts**: Transfer-based adversarial attacks can evaluate model robustness in the black-box setting. Several methods have demonstrated impressive untargeted transferability, however, it is still challenging to efficiently produce targeted transferability. To this end, we develop a simple yet effective framework to craft targeted transfer-based adversarial examples, applying a hierarchical generative network. In particular, we contribute to amortized designs that well adapt to multi-class targeted attacks. Extensive experiments on ImageNet show that our method improves the success rates of targeted black-box attacks by a significant margin over the existing methods -- it reaches an average success rate of 29.1\% against six diverse models based only on one substitute white-box model, which significantly outperforms the state-of-the-art gradient-based attack methods. Moreover, the proposed method is also more efficient beyond an order of magnitude than gradient-based methods.



## **20. Synthetic Dataset Generation for Adversarial Machine Learning Research**

cs.CV

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10719v1)

**Authors**: Xiruo Liu, Shibani Singh, Cory Cornelius, Colin Busho, Mike Tan, Anindya Paul, Jason Martin

**Abstracts**: Existing adversarial example research focuses on digitally inserted perturbations on top of existing natural image datasets. This construction of adversarial examples is not realistic because it may be difficult, or even impossible, for an attacker to deploy such an attack in the real-world due to sensing and environmental effects. To better understand adversarial examples against cyber-physical systems, we propose approximating the real-world through simulation. In this paper we describe our synthetic dataset generation tool that enables scalable collection of such a synthetic dataset with realistic adversarial examples. We use the CARLA simulator to collect such a dataset and demonstrate simulated attacks that undergo the same environmental transforms and processing as real-world images. Our tools have been used to collect datasets to help evaluate the efficacy of adversarial examples, and can be found at https://github.com/carla-simulator/carla/pull/4992.



## **21. Careful What You Wish For: on the Extraction of Adversarially Trained Models**

cs.LG

To be published in the proceedings of the 19th Annual International  Conference on Privacy, Security & Trust (PST 2022). The conference  proceedings will be included in IEEE Xplore as in previous editions of the  conference

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10561v1)

**Authors**: Kacem Khaled, Gabriela Nicolescu, Felipe Gohring de Magalhães

**Abstracts**: Recent attacks on Machine Learning (ML) models such as evasion attacks with adversarial examples and models stealing through extraction attacks pose several security and privacy threats. Prior work proposes to use adversarial training to secure models from adversarial examples that can evade the classification of a model and deteriorate its performance. However, this protection technique affects the model's decision boundary and its prediction probabilities, hence it might raise model privacy risks. In fact, a malicious user using only a query access to the prediction output of a model can extract it and obtain a high-accuracy and high-fidelity surrogate model. To have a greater extraction, these attacks leverage the prediction probabilities of the victim model. Indeed, all previous work on extraction attacks do not take into consideration the changes in the training process for security purposes. In this paper, we propose a framework to assess extraction attacks on adversarially trained models with vision datasets. To the best of our knowledge, our work is the first to perform such evaluation. Through an extensive empirical study, we demonstrate that adversarially trained models are more vulnerable to extraction attacks than models obtained under natural training circumstances. They can achieve up to $\times1.2$ higher accuracy and agreement with a fraction lower than $\times0.75$ of the queries. We additionally find that the adversarial robustness capability is transferable through extraction attacks, i.e., extracted Deep Neural Networks (DNNs) from robust models show an enhanced accuracy to adversarial examples compared to extracted DNNs from naturally trained (i.e. standard) models.



## **22. Triangle Attack: A Query-efficient Decision-based Adversarial Attack**

cs.CV

Accepted by ECCV 2022, code is available at  https://github.com/xiaosen-wang/TA

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2112.06569v3)

**Authors**: Xiaosen Wang, Zeliang Zhang, Kangheng Tong, Dihong Gong, Kun He, Zhifeng Li, Wei Liu

**Abstracts**: Decision-based attack poses a severe threat to real-world applications since it regards the target model as a black box and only accesses the hard prediction label. Great efforts have been made recently to decrease the number of queries; however, existing decision-based attacks still require thousands of queries in order to generate good quality adversarial examples. In this work, we find that a benign sample, the current and the next adversarial examples can naturally construct a triangle in a subspace for any iterative attacks. Based on the law of sines, we propose a novel Triangle Attack (TA) to optimize the perturbation by utilizing the geometric information that the longer side is always opposite the larger angle in any triangle. However, directly applying such information on the input image is ineffective because it cannot thoroughly explore the neighborhood of the input sample in the high dimensional space. To address this issue, TA optimizes the perturbation in the low frequency space for effective dimensionality reduction owing to the generality of such geometric property. Extensive evaluations on ImageNet dataset show that TA achieves a much higher attack success rate within 1,000 queries and needs a much less number of queries to achieve the same attack success rate under various perturbation budgets than existing decision-based attacks. With such high efficiency, we further validate the applicability of TA on real-world API, i.e., Tencent Cloud API.



## **23. Knowledge-enhanced Black-box Attacks for Recommendations**

cs.LG

Accepted in the KDD'22

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10307v1)

**Authors**: Jingfan Chen, Wenqi Fan, Guanghui Zhu, Xiangyu Zhao, Chunfeng Yuan, Qing Li, Yihua Huang

**Abstracts**: Recent studies have shown that deep neural networks-based recommender systems are vulnerable to adversarial attacks, where attackers can inject carefully crafted fake user profiles (i.e., a set of items that fake users have interacted with) into a target recommender system to achieve malicious purposes, such as promote or demote a set of target items. Due to the security and privacy concerns, it is more practical to perform adversarial attacks under the black-box setting, where the architecture/parameters and training data of target systems cannot be easily accessed by attackers. However, generating high-quality fake user profiles under black-box setting is rather challenging with limited resources to target systems. To address this challenge, in this work, we introduce a novel strategy by leveraging items' attribute information (i.e., items' knowledge graph), which can be publicly accessible and provide rich auxiliary knowledge to enhance the generation of fake user profiles. More specifically, we propose a knowledge graph-enhanced black-box attacking framework (KGAttack) to effectively learn attacking policies through deep reinforcement learning techniques, in which knowledge graph is seamlessly integrated into hierarchical policy networks to generate fake user profiles for performing adversarial black-box attacks. Comprehensive experiments on various real-world datasets demonstrate the effectiveness of the proposed attacking framework under the black-box setting.



## **24. Image Generation Network for Covert Transmission in Online Social Network**

cs.CV

ACMMM2022 Poster

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10292v1)

**Authors**: Zhengxin You, Qichao Ying, Sheng Li, Zhenxing Qian, Xinpeng Zhang

**Abstracts**: Online social networks have stimulated communications over the Internet more than ever, making it possible for secret message transmission over such noisy channels. In this paper, we propose a Coverless Image Steganography Network, called CIS-Net, that synthesizes a high-quality image directly conditioned on the secret message to transfer. CIS-Net is composed of four modules, namely, the Generation, Adversarial, Extraction, and Noise Module. The receiver can extract the hidden message without any loss even the images have been distorted by JPEG compression attacks. To disguise the behaviour of steganography, we collected images in the context of profile photos and stickers and train our network accordingly. As such, the generated images are more inclined to escape from malicious detection and attack. The distinctions from previous image steganography methods are majorly the robustness and losslessness against diverse attacks. Experiments over diverse public datasets have manifested the superior ability of anti-steganalysis.



## **25. Switching One-Versus-the-Rest Loss to Increase the Margin of Logits for Adversarial Robustness**

cs.LG

20 pages, 16 figures

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10283v1)

**Authors**: Sekitoshi Kanai, Shin'ya Yamaguchi, Masanori Yamada, Hiroshi Takahashi, Yasutoshi Ida

**Abstracts**: Defending deep neural networks against adversarial examples is a key challenge for AI safety. To improve the robustness effectively, recent methods focus on important data points near the decision boundary in adversarial training. However, these methods are vulnerable to Auto-Attack, which is an ensemble of parameter-free attacks for reliable evaluation. In this paper, we experimentally investigate the causes of their vulnerability and find that existing methods reduce margins between logits for the true label and the other labels while keeping their gradient norms non-small values. Reduced margins and non-small gradient norms cause their vulnerability since the largest logit can be easily flipped by the perturbation. Our experiments also show that the histogram of the logit margins has two peaks, i.e., small and large logit margins. From the observations, we propose switching one-versus-the-rest loss (SOVR), which uses one-versus-the-rest loss when data have small logit margins so that it increases the margins. We find that SOVR increases logit margins more than existing methods while keeping gradient norms small and outperforms them in terms of the robustness against Auto-Attack.



## **26. FOCUS: Fairness via Agent-Awareness for Federated Learning on Heterogeneous Data**

cs.LG

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10265v1)

**Authors**: Wenda Chu, Chulin Xie, Boxin Wang, Linyi Li, Lang Yin, Han Zhao, Bo Li

**Abstracts**: Federated learning (FL) provides an effective paradigm to train machine learning models over distributed data with privacy protection. However, recent studies show that FL is subject to various security, privacy, and fairness threats due to the potentially malicious and heterogeneous local agents. For instance, it is vulnerable to local adversarial agents who only contribute low-quality data, with the goal of harming the performance of those with high-quality data. This kind of attack hence breaks existing definitions of fairness in FL that mainly focus on a certain notion of performance parity. In this work, we aim to address this limitation and propose a formal definition of fairness via agent-awareness for FL (FAA), which takes the heterogeneous data contributions of local agents into account. In addition, we propose a fair FL training algorithm based on agent clustering (FOCUS) to achieve FAA. Theoretically, we prove the convergence and optimality of FOCUS under mild conditions for linear models and general convex loss functions with bounded smoothness. We also prove that FOCUS always achieves higher fairness measured by FAA compared with standard FedAvg protocol under both linear models and general convex loss functions. Empirically, we evaluate FOCUS on four datasets, including synthetic data, images, and texts under different settings, and we show that FOCUS achieves significantly higher fairness based on FAA while maintaining similar or even higher prediction accuracy compared with FedAvg.



## **27. Illusionary Attacks on Sequential Decision Makers and Countermeasures**

cs.AI

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2207.10170v1)

**Authors**: Tim Franzmeyer, João F. Henriques, Jakob N. Foerster, Philip H. S. Torr, Adel Bibi, Christian Schroeder de Witt

**Abstracts**: Autonomous intelligent agents deployed to the real-world need to be robust against adversarial attacks on sensory inputs. Existing work in reinforcement learning focuses on minimum-norm perturbation attacks, which were originally introduced to mimic a notion of perceptual invariance in computer vision. In this paper, we note that such minimum-norm perturbation attacks can be trivially detected by victim agents, as these result in observation sequences that are not consistent with the victim agent's actions. Furthermore, many real-world agents, such as physical robots, commonly operate under human supervisors, which are not susceptible to such perturbation attacks. As a result, we propose to instead focus on illusionary attacks, a novel form of attack that is consistent with the world model of the victim agent. We provide a formal definition of this novel attack framework, explore its characteristics under a variety of conditions, and conclude that agents must seek realism feedback to be robust to illusionary attacks.



## **28. PFMC: a parallel symbolic model checker for security protocol verification**

cs.LO

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2207.09895v1)

**Authors**: Alex James, Alwen Tiu, Nisansala Yatapanage

**Abstracts**: We present an investigation into the design and implementation of a parallel model checker for security protocol verification that is based on a symbolic model of the adversary, where instantiations of concrete terms and messages are avoided until needed to resolve a particular assertion. We propose to build on this naturally lazy approach to parallelise this symbolic state exploration and evaluation. We utilise the concept of strategies in Haskell, which abstracts away from the low-level details of thread management and modularly adds parallel evaluation strategies (encapsulated as a monad in Haskell). We build on an existing symbolic model checker, OFMC, which is already implemented in Haskell. We show that there is a very significant speed up of around 3-5 times improvement when moving from the original single-threaded implementation of OFMC to our multi-threaded version, for both the Dolev-Yao attacker model and more general algebraic attacker models. We identify several issues in parallelising the model checker: among others, controlling growth of memory consumption, balancing lazy vs strict evaluation, and achieving an optimal granularity of parallelism.



## **29. Adaptive Image Transformations for Transfer-based Adversarial Attack**

cs.CV

34 pages, 7 figures, 11 tables. Accepted by ECCV2022

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2111.13844v3)

**Authors**: Zheng Yuan, Jie Zhang, Shiguang Shan

**Abstracts**: Adversarial attacks provide a good way to study the robustness of deep learning models. One category of methods in transfer-based black-box attack utilizes several image transformation operations to improve the transferability of adversarial examples, which is effective, but fails to take the specific characteristic of the input image into consideration. In this work, we propose a novel architecture, called Adaptive Image Transformation Learner (AITL), which incorporates different image transformation operations into a unified framework to further improve the transferability of adversarial examples. Unlike the fixed combinational transformations used in existing works, our elaborately designed transformation learner adaptively selects the most effective combination of image transformations specific to the input image. Extensive experiments on ImageNet demonstrate that our method significantly improves the attack success rates on both normally trained models and defense models under various settings.



## **30. On the Robustness of Quality Measures for GANs**

cs.LG

Accepted at the European Conference in Computer Vision (ECCV 2022)

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2201.13019v2)

**Authors**: Motasem Alfarra, Juan C. Pérez, Anna Frühstück, Philip H. S. Torr, Peter Wonka, Bernard Ghanem

**Abstracts**: This work evaluates the robustness of quality measures of generative models such as Inception Score (IS) and Fr\'echet Inception Distance (FID). Analogous to the vulnerability of deep models against a variety of adversarial attacks, we show that such metrics can also be manipulated by additive pixel perturbations. Our experiments indicate that one can generate a distribution of images with very high scores but low perceptual quality. Conversely, one can optimize for small imperceptible perturbations that, when added to real world images, deteriorate their scores. We further extend our evaluation to generative models themselves, including the state of the art network StyleGANv2. We show the vulnerability of both the generative model and the FID against additive perturbations in the latent space. Finally, we show that the FID can be robustified by simply replacing the standard Inception with a robust Inception. We validate the effectiveness of the robustified metric through extensive experiments, showing it is more robust against manipulation.



## **31. On the Versatile Uses of Partial Distance Correlation in Deep Learning**

cs.CV

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2207.09684v1)

**Authors**: Xingjian Zhen, Zihang Meng, Rudrasis Chakraborty, Vikas Singh

**Abstracts**: Comparing the functional behavior of neural network models, whether it is a single network over time or two (or more networks) during or post-training, is an essential step in understanding what they are learning (and what they are not), and for identifying strategies for regularization or efficiency improvements. Despite recent progress, e.g., comparing vision transformers to CNNs, systematic comparison of function, especially across different networks, remains difficult and is often carried out layer by layer. Approaches such as canonical correlation analysis (CCA) are applicable in principle, but have been sparingly used so far. In this paper, we revisit a (less widely known) from statistics, called distance correlation (and its partial variant), designed to evaluate correlation between feature spaces of different dimensions. We describe the steps necessary to carry out its deployment for large scale models -- this opens the door to a surprising array of applications ranging from conditioning one deep model w.r.t. another, learning disentangled representations as well as optimizing diverse models that would directly be more robust to adversarial attacks. Our experiments suggest a versatile regularizer (or constraint) with many advantages, which avoids some of the common difficulties one faces in such analyses. Code is at https://github.com/zhenxingjian/Partial_Distance_Correlation.



## **32. Detecting Textual Adversarial Examples through Randomized Substitution and Vote**

cs.CL

Accepted by UAI 2022, code is avaliable at  https://github.com/JHL-HUST/RSV

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2109.05698v2)

**Authors**: Xiaosen Wang, Yifeng Xiong, Kun He

**Abstracts**: A line of work has shown that natural text processing models are vulnerable to adversarial examples. Correspondingly, various defense methods are proposed to mitigate the threat of textual adversarial examples, eg, adversarial training, input transformations, detection, etc. In this work, we treat the optimization process for synonym substitution based textual adversarial attacks as a specific sequence of word replacement, in which each word mutually influences other words. We identify that we could destroy such mutual interaction and eliminate the adversarial perturbation by randomly substituting a word with its synonyms. Based on this observation, we propose a novel textual adversarial example detection method, termed Randomized Substitution and Vote (RS&V), which votes the prediction label by accumulating the logits of k samples generated by randomly substituting the words in the input text with synonyms. The proposed RS&V is generally applicable to any existing neural networks without modification on the architecture or extra training, and it is orthogonal to prior work on making the classification network itself more robust. Empirical evaluations on three benchmark datasets demonstrate that our RS&V could detect the textual adversarial examples more successfully than the existing detection methods while maintaining the high classification accuracy on benign samples.



## **33. Diversified Adversarial Attacks based on Conjugate Gradient Method**

cs.LG

Proceedings of the 39th International Conference on Machine Learning  (ICML 2022)

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2206.09628v2)

**Authors**: Keiichiro Yamamura, Haruki Sato, Nariaki Tateiwa, Nozomi Hata, Toru Mitsutake, Issa Oe, Hiroki Ishikura, Katsuki Fujisawa

**Abstracts**: Deep learning models are vulnerable to adversarial examples, and adversarial attacks used to generate such examples have attracted considerable research interest. Although existing methods based on the steepest descent have achieved high attack success rates, ill-conditioned problems occasionally reduce their performance. To address this limitation, we utilize the conjugate gradient (CG) method, which is effective for this type of problem, and propose a novel attack algorithm inspired by the CG method, named the Auto Conjugate Gradient (ACG) attack. The results of large-scale evaluation experiments conducted on the latest robust models show that, for most models, ACG was able to find more adversarial examples with fewer iterations than the existing SOTA algorithm Auto-PGD (APGD). We investigated the difference in search performance between ACG and APGD in terms of diversification and intensification, and define a measure called Diversity Index (DI) to quantify the degree of diversity. From the analysis of the diversity using this index, we show that the more diverse search of the proposed method remarkably improves its attack success rate.



## **34. Towards Robust Multivariate Time-Series Forecasting: Adversarial Attacks and Defense Mechanisms**

cs.LG

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09572v1)

**Authors**: Linbo Liu, Youngsuk Park, Trong Nghia Hoang, Hilaf Hasson, Jun Huan

**Abstracts**: As deep learning models have gradually become the main workhorse of time series forecasting, the potential vulnerability under adversarial attacks to forecasting and decision system accordingly has emerged as a main issue in recent years. Albeit such behaviors and defense mechanisms started to be investigated for the univariate time series forecasting, there are still few studies regarding the multivariate forecasting which is often preferred due to its capacity to encode correlations between different time series. In this work, we study and design adversarial attack on multivariate probabilistic forecasting models, taking into consideration attack budget constraints and the correlation architecture between multiple time series. Specifically, we investigate a sparse indirect attack that hurts the prediction of an item (time series) by only attacking the history of a small number of other items to save attacking cost. In order to combat these attacks, we also develop two defense strategies. First, we adopt randomized smoothing to multivariate time series scenario and verify its effectiveness via empirical experiments. Second, we leverage a sparse attacker to enable end-to-end adversarial training that delivers robust probabilistic forecasters. Extensive experiments on real dataset confirm that our attack schemes are powerful and our defend algorithms are more effective compared with other baseline defense mechanisms.



## **35. Increasing the Cost of Model Extraction with Calibrated Proof of Work**

cs.CR

Published as a conference paper at ICLR 2022 (Spotlight - 5% of  submitted papers)

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2201.09243v2)

**Authors**: Adam Dziedzic, Muhammad Ahmad Kaleem, Yu Shen Lu, Nicolas Papernot

**Abstracts**: In model extraction attacks, adversaries can steal a machine learning model exposed via a public API by repeatedly querying it and adjusting their own model based on obtained predictions. To prevent model stealing, existing defenses focus on detecting malicious queries, truncating, or distorting outputs, thus necessarily introducing a tradeoff between robustness and model utility for legitimate users. Instead, we propose to impede model extraction by requiring users to complete a proof-of-work before they can read the model's predictions. This deters attackers by greatly increasing (even up to 100x) the computational effort needed to leverage query access for model extraction. Since we calibrate the effort required to complete the proof-of-work to each query, this only introduces a slight overhead for regular users (up to 2x). To achieve this, our calibration applies tools from differential privacy to measure the information revealed by a query. Our method requires no modification of the victim model and can be applied by machine learning practitioners to guard their publicly exposed models against being easily stolen.



## **36. Assaying Out-Of-Distribution Generalization in Transfer Learning**

cs.LG

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09239v1)

**Authors**: Florian Wenzel, Andrea Dittadi, Peter Vincent Gehler, Carl-Johann Simon-Gabriel, Max Horn, Dominik Zietlow, David Kernert, Chris Russell, Thomas Brox, Bernt Schiele, Bernhard Schölkopf, Francesco Locatello

**Abstracts**: Since out-of-distribution generalization is a generally ill-posed problem, various proxy targets (e.g., calibration, adversarial robustness, algorithmic corruptions, invariance across shifts) were studied across different research programs resulting in different recommendations. While sharing the same aspirational goal, these approaches have never been tested under the same experimental conditions on real data. In this paper, we take a unified view of previous work, highlighting message discrepancies that we address empirically, and providing recommendations on how to measure the robustness of a model and how to improve it. To this end, we collect 172 publicly available dataset pairs for training and out-of-distribution evaluation of accuracy, calibration error, adversarial attacks, environment invariance, and synthetic corruptions. We fine-tune over 31k networks, from nine different architectures in the many- and few-shot setting. Our findings confirm that in- and out-of-distribution accuracies tend to increase jointly, but show that their relation is largely dataset-dependent, and in general more nuanced and more complex than posited by previous, smaller scale studies.



## **37. MUD-PQFed: Towards Malicious User Detection in Privacy-Preserving Quantized Federated Learning**

cs.CR

13 pages,13 figures

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09080v1)

**Authors**: Hua Ma, Qun Li, Yifeng Zheng, Zhi Zhang, Xiaoning Liu, Yansong Gao, Said F. Al-Sarawi, Derek Abbott

**Abstracts**: Federated Learning (FL), a distributed machine learning paradigm, has been adapted to mitigate privacy concerns for customers. Despite their appeal, there are various inference attacks that can exploit shared-plaintext model updates to embed traces of customer private information, leading to serious privacy concerns. To alleviate this privacy issue, cryptographic techniques such as Secure Multi-Party Computation and Homomorphic Encryption have been used for privacy-preserving FL. However, such security issues in privacy-preserving FL are poorly elucidated and underexplored. This work is the first attempt to elucidate the triviality of performing model corruption attacks on privacy-preserving FL based on lightweight secret sharing. We consider scenarios in which model updates are quantized to reduce communication overhead in this case, where an adversary can simply provide local parameters outside the legal range to corrupt the model. We then propose the MUD-PQFed protocol, which can precisely detect malicious clients performing attacks and enforce fair penalties. By removing the contributions of detected malicious clients, the global model utility is preserved to be comparable to the baseline global model without the attack. Extensive experiments validate effectiveness in maintaining baseline accuracy and detecting malicious clients in a fine-grained manner



## **38. $\ell_\infty$-Robustness and Beyond: Unleashing Efficient Adversarial Training**

cs.LG

Accepted to the 17th European Conference on Computer Vision (ECCV  2022)

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2112.00378v2)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstracts**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches in training robust models against such attacks. However, it is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration, hampering its effectiveness. Recently, Fast Adversarial Training (FAT) was proposed that can obtain robust models efficiently. However, the reasons behind its success are not fully understood, and more importantly, it can only train robust models for $\ell_\infty$-bounded attacks as it uses FGSM during training. In this paper, by leveraging the theory of coreset selection, we show how selecting a small subset of training data provides a general, more principled approach toward reducing the time complexity of robust training. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training (PAT). Our experimental results indicate that our approach speeds up adversarial training by 2-3 times while experiencing a slight reduction in the clean and robust accuracy.



## **39. Decorrelative Network Architecture for Robust Electrocardiogram Classification**

cs.LG

12 pages, 6 figures

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09031v1)

**Authors**: Christopher Wiedeman, Ge Wang

**Abstracts**: Artificial intelligence has made great progresses in medical data analysis, but the lack of robustness and interpretability has kept these methods from being widely deployed. In particular, data-driven models are vulnerable to adversarial attacks, which are small, targeted perturbations that dramatically degrade model performance. As a recent example, while deep learning has shown impressive performance in electrocardiogram (ECG) classification, Han et al. crafted realistic perturbations that fooled the network 74% of the time [2020]. Current adversarial defense paradigms are computationally intensive and impractical for many high dimensional problems. Previous research indicates that a network vulnerability is related to the features learned during training. We propose a novel approach based on ensemble decorrelation and Fourier partitioning for training parallel network arms into a decorrelated architecture to learn complementary features, significantly reducing the chance of a perturbation fooling all arms of the deep learning model. We test our approach in ECG classification, demonstrating a much-improved 77.2% chance of at least one correct network arm on the strongest adversarial attack tested, in contrast to a 21.7% chance from a comparable ensemble. Our approach does not require expensive optimization with adversarial samples, and thus can be scaled to large problems. These methods can easily be applied to other tasks for improved network robustness.



## **40. Defending Substitution-Based Profile Pollution Attacks on Sequential Recommenders**

cs.IR

Accepted to RecSys 2022

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.11237v1)

**Authors**: Zhenrui Yue, Huimin Zeng, Ziyi Kou, Lanyu Shang, Dong Wang

**Abstracts**: While sequential recommender systems achieve significant improvements on capturing user dynamics, we argue that sequential recommenders are vulnerable against substitution-based profile pollution attacks. To demonstrate our hypothesis, we propose a substitution-based adversarial attack algorithm, which modifies the input sequence by selecting certain vulnerable elements and substituting them with adversarial items. In both untargeted and targeted attack scenarios, we observe significant performance deterioration using the proposed profile pollution algorithm. Motivated by such observations, we design an efficient adversarial defense method called Dirichlet neighborhood sampling. Specifically, we sample item embeddings from a convex hull constructed by multi-hop neighbors to replace the original items in input sequences. During sampling, a Dirichlet distribution is used to approximate the probability distribution in the neighborhood such that the recommender learns to combat local perturbations. Additionally, we design an adversarial training method tailored for sequential recommender systems. In particular, we represent selected items with one-hot encodings and perform gradient ascent on the encodings to search for the worst case linear combination of item embeddings in training. As such, the embedding function learns robust item representations and the trained recommender is resistant to test-time adversarial examples. Extensive experiments show the effectiveness of both our attack and defense methods, which consistently outperform baselines by a significant margin across model architectures and datasets.



## **41. Multi-step domain adaptation by adversarial attack to $\mathcal{H} Δ\mathcal{H}$-divergence**

cs.LG

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08948v1)

**Authors**: Arip Asadulaev, Alexander Panfilov, Andrey Filchenkov

**Abstracts**: Adversarial examples are transferable between different models. In our paper, we propose to use this property for multi-step domain adaptation. In unsupervised domain adaptation settings, we demonstrate that replacing the source domain with adversarial examples to $\mathcal{H} \Delta \mathcal{H}$-divergence can improve source classifier accuracy on the target domain. Our method can be connected to most domain adaptation techniques. We conducted a range of experiments and achieved improvement in accuracy on Digits and Office-Home datasets.



## **42. Benchmarking Machine Learning Robustness in Covid-19 Genome Sequence Classification**

q-bio.GN

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08898v1)

**Authors**: Sarwan Ali, Bikram Sahoo, Alexander Zelikovskiy, Pin-Yu Chen, Murray Patterson

**Abstracts**: The rapid spread of the COVID-19 pandemic has resulted in an unprecedented amount of sequence data of the SARS-CoV-2 genome -- millions of sequences and counting. This amount of data, while being orders of magnitude beyond the capacity of traditional approaches to understanding the diversity, dynamics, and evolution of viruses is nonetheless a rich resource for machine learning (ML) approaches as alternatives for extracting such important information from these data. It is of hence utmost importance to design a framework for testing and benchmarking the robustness of these ML models.   This paper makes the first effort (to our knowledge) to benchmark the robustness of ML models by simulating biological sequences with errors. In this paper, we introduce several ways to perturb SARS-CoV-2 genome sequences to mimic the error profiles of common sequencing platforms such as Illumina and PacBio. We show from experiments on a wide array of ML models that some simulation-based approaches are more robust (and accurate) than others for specific embedding methods to certain adversarial attacks to the input sequences. Our benchmarking framework may assist researchers in properly assessing different ML models and help them understand the behavior of the SARS-CoV-2 virus or avoid possible future pandemics.



## **43. Prior-Guided Adversarial Initialization for Fast Adversarial Training**

cs.CV

ECCV 2022

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08859v1)

**Authors**: Xiaojun Jia, Yong Zhang, Xingxing Wei, Baoyuan Wu, Ke Ma, Jue Wang, Xiaochun Cao

**Abstracts**: Fast adversarial training (FAT) effectively improves the efficiency of standard adversarial training (SAT). However, initial FAT encounters catastrophic overfitting, i.e.,the robust accuracy against adversarial attacks suddenly and dramatically decreases. Though several FAT variants spare no effort to prevent overfitting, they sacrifice much calculation cost. In this paper, we explore the difference between the training processes of SAT and FAT and observe that the attack success rate of adversarial examples (AEs) of FAT gets worse gradually in the late training stage, resulting in overfitting. The AEs are generated by the fast gradient sign method (FGSM) with a zero or random initialization. Based on the observation, we propose a prior-guided FGSM initialization method to avoid overfitting after investigating several initialization strategies, improving the quality of the AEs during the whole training process. The initialization is formed by leveraging historically generated AEs without additional calculation cost. We further provide a theoretical analysis for the proposed initialization method. We also propose a simple yet effective regularizer based on the prior-guided initialization,i.e., the currently generated perturbation should not deviate too much from the prior-guided initialization. The regularizer adopts both historical and current adversarial perturbations to guide the model learning. Evaluations on four datasets demonstrate that the proposed method can prevent catastrophic overfitting and outperform state-of-the-art FAT methods. The code is released at https://github.com/jiaxiaojunQAQ/FGSM-PGI.



## **44. Adversarial Pixel Restoration as a Pretext Task for Transferable Perturbations**

cs.CV

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08803v1)

**Authors**: Hashmat Shadab Malik, Shahina K Kunhimon, Muzammal Naseer, Salman Khan, Fahad Shahbaz Khan

**Abstracts**: Transferable adversarial attacks optimize adversaries from a pretrained surrogate model and known label space to fool the unknown black-box models. Therefore, these attacks are restricted by the availability of an effective surrogate model. In this work, we relax this assumption and propose Adversarial Pixel Restoration as a self-supervised alternative to train an effective surrogate model from scratch under the condition of no labels and few data samples. Our training approach is based on a min-max objective which reduces overfitting via an adversarial objective and thus optimizes for a more generalizable surrogate model. Our proposed attack is complimentary to our adversarial pixel restoration and is independent of any task specific objective as it can be launched in a self-supervised manner. We successfully demonstrate the adversarial transferability of our approach to Vision Transformers as well as Convolutional Neural Networks for the tasks of classification, object detection, and video segmentation. Our codes & pre-trained surrogate models are available at: https://github.com/HashmatShadab/APR



## **45. Are Vision Transformers Robust to Patch Perturbations?**

cs.CV

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2111.10659v2)

**Authors**: Jindong Gu, Volker Tresp, Yao Qin

**Abstracts**: Recent advances in Vision Transformer (ViT) have demonstrated its impressive performance in image classification, which makes it a promising alternative to Convolutional Neural Network (CNN). Unlike CNNs, ViT represents an input image as a sequence of image patches. The patch-based input image representation makes the following question interesting: How does ViT perform when individual input image patches are perturbed with natural corruptions or adversarial perturbations, compared to CNNs? In this work, we study the robustness of ViT to patch-wise perturbations. Surprisingly, we find that ViTs are more robust to naturally corrupted patches than CNNs, whereas they are more vulnerable to adversarial patches. Furthermore, we discover that the attention mechanism greatly affects the robustness of vision transformers. Specifically, the attention module can help improve the robustness of ViT by effectively ignoring natural corrupted patches. However, when ViTs are attacked by an adversary, the attention mechanism can be easily fooled to focus more on the adversarially perturbed patches and cause a mistake. Based on our analysis, we propose a simple temperature-scaling based method to improve the robustness of ViT against adversarial patches. Extensive qualitative and quantitative experiments are performed to support our findings, understanding, and improvement of ViT robustness to patch-wise perturbations across a set of transformer-based architectures.



## **46. Authentication Attacks on Projection-based Cancelable Biometric Schemes**

cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2110.15163v6)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.



## **47. Detection of Poisoning Attacks with Anomaly Detection in Federated Learning for Healthcare Applications: A Machine Learning Approach**

cs.LG

We will updated this article soon

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08486v1)

**Authors**: Ali Raza, Shujun Li, Kim-Phuc Tran, Ludovic Koehl

**Abstracts**: The application of Federated Learning (FL) is steadily increasing, especially in privacy-aware applications, such as healthcare. However, its applications have been limited by security concerns due to various adversarial attacks, such as poisoning attacks (model and data poisoning). Such attacks attempt to poison the local models and data to manipulate the global models in order to obtain undue benefits and malicious use. Traditional methods of data auditing to mitigate poisoning attacks find their limited applications in FL because the edge devices never share their raw data directly due to privacy concerns, and are globally distributed with no insight into their training data. Thereafter, it is challenging to develop appropriate strategies to address such attacks and minimize their impact on the global model in federated learning. In order to address such challenges in FL, we proposed a novel framework to detect poisoning attacks using deep neural networks and support vector machines, in the form of anomaly without acquiring any direct access or information about the underlying training data of local edge devices. We illustrate and evaluate the proposed framework using different state of art poisoning attacks for two different healthcare applications: Electrocardiograph classification and human activity recognition. Our experimental analysis shows that the proposed method can efficiently detect poisoning attacks and can remove the identified poisoned updated from the global aggregation. Thereafter can increase the performance of the federated global.



## **48. Towards Automated Classification of Attackers' TTPs by combining NLP with ML Techniques**

cs.CR

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08478v1)

**Authors**: Clemens Sauerwein, Alexander Pfohl

**Abstracts**: The increasingly sophisticated and growing number of threat actors along with the sheer speed at which cyber attacks unfold, make timely identification of attacks imperative to an organisations' security. Consequently, persons responsible for security employ a large variety of information sources concerning emerging attacks, attackers' course of actions or indicators of compromise. However, a vast amount of the needed security information is available in unstructured textual form, which complicates the automated and timely extraction of attackers' Tactics, Techniques and Procedures (TTPs). In order to address this problem we systematically evaluate and compare different Natural Language Processing (NLP) and machine learning techniques used for security information extraction in research. Based on our investigations we propose a data processing pipeline that automatically classifies unstructured text according to attackers' tactics and techniques derived from a knowledge base of adversary tactics, techniques and procedures.



## **49. A Perturbation-Constrained Adversarial Attack for Evaluating the Robustness of Optical Flow**

cs.CV

Accepted at the European Conference on Computer Vision (ECCV) 2022

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2203.13214v2)

**Authors**: Jenny Schmalfuss, Philipp Scholze, Andrés Bruhn

**Abstracts**: Recent optical flow methods are almost exclusively judged in terms of accuracy, while their robustness is often neglected. Although adversarial attacks offer a useful tool to perform such an analysis, current attacks on optical flow methods focus on real-world attacking scenarios rather than a worst case robustness assessment. Hence, in this work, we propose a novel adversarial attack - the Perturbation-Constrained Flow Attack (PCFA) - that emphasizes destructivity over applicability as a real-world attack. PCFA is a global attack that optimizes adversarial perturbations to shift the predicted flow towards a specified target flow, while keeping the L2 norm of the perturbation below a chosen bound. Our experiments demonstrate PCFA's applicability in white- and black-box settings, and show it finds stronger adversarial samples than previous attacks. Based on these strong samples, we provide the first joint ranking of optical flow methods considering both prediction quality and adversarial robustness, which reveals state-of-the-art methods to be particularly vulnerable. Code is available at https://github.com/cv-stuttgart/PCFA.



## **50. Universal Adversarial Examples in Remote Sensing: Methodology and Benchmark**

cs.CV

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2202.07054v3)

**Authors**: Yonghao Xu, Pedram Ghamisi

**Abstracts**: Deep neural networks have achieved great success in many important remote sensing tasks. Nevertheless, their vulnerability to adversarial examples should not be neglected. In this study, we systematically analyze the universal adversarial examples in remote sensing data for the first time, without any knowledge from the victim model. Specifically, we propose a novel black-box adversarial attack method, namely Mixup-Attack, and its simple variant Mixcut-Attack, for remote sensing data. The key idea of the proposed methods is to find common vulnerabilities among different networks by attacking the features in the shallow layer of a given surrogate model. Despite their simplicity, the proposed methods can generate transferable adversarial examples that deceive most of the state-of-the-art deep neural networks in both scene classification and semantic segmentation tasks with high success rates. We further provide the generated universal adversarial examples in the dataset named UAE-RS, which is the first dataset that provides black-box adversarial samples in the remote sensing field. We hope UAE-RS may serve as a benchmark that helps researchers to design deep neural networks with strong resistance toward adversarial attacks in the remote sensing field. Codes and the UAE-RS dataset are available online (https://github.com/YonghaoXu/UAE-RS).



