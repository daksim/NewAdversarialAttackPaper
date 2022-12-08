# Latest Adversarial Attack Papers
**update at 2022-12-08 20:52:13**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Multiple Perturbation Attack: Attack Pixelwise Under Different $\ell_p$-norms For Better Adversarial Performance**

cs.CV

18 pages, 8 figures, 7 tables

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2212.03069v2) [paper-pdf](http://arxiv.org/pdf/2212.03069v2)

**Authors**: Ngoc N. Tran, Anh Tuan Bui, Dinh Phung, Trung Le

**Abstract**: Adversarial machine learning has been both a major concern and a hot topic recently, especially with the ubiquitous use of deep neural networks in the current landscape. Adversarial attacks and defenses are usually likened to a cat-and-mouse game in which defenders and attackers evolve over the time. On one hand, the goal is to develop strong and robust deep networks that are resistant to malicious actors. On the other hand, in order to achieve that, we need to devise even stronger adversarial attacks to challenge these defense models. Most of existing attacks employs a single $\ell_p$ distance (commonly, $p\in\{1,2,\infty\}$) to define the concept of closeness and performs steepest gradient ascent w.r.t. this $p$-norm to update all pixels in an adversarial example in the same way. These $\ell_p$ attacks each has its own pros and cons; and there is no single attack that can successfully break through defense models that are robust against multiple $\ell_p$ norms simultaneously. Motivated by these observations, we come up with a natural approach: combining various $\ell_p$ gradient projections on a pixel level to achieve a joint adversarial perturbation. Specifically, we learn how to perturb each pixel to maximize the attack performance, while maintaining the overall visual imperceptibility of adversarial examples. Finally, through various experiments with standardized benchmarks, we show that our method outperforms most current strong attacks across state-of-the-art defense mechanisms, while retaining its ability to remain clean visually.



## **2. Universal Backdoor Attacks Detection via Adaptive Adversarial Probe**

cs.CV

8 pages, 8 figures

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2209.05244v3) [paper-pdf](http://arxiv.org/pdf/2209.05244v3)

**Authors**: Yuhang Wang, Huafeng Shi, Rui Min, Ruijia Wu, Siyuan Liang, Yichao Wu, Ding Liang, Aishan Liu

**Abstract**: Extensive evidence has demonstrated that deep neural networks (DNNs) are vulnerable to backdoor attacks, which motivates the development of backdoor attacks detection. Most detection methods are designed to verify whether a model is infected with presumed types of backdoor attacks, yet the adversary is likely to generate diverse backdoor attacks in practice that are unforeseen to defenders, which challenge current detection strategies. In this paper, we focus on this more challenging scenario and propose a universal backdoor attacks detection method named Adaptive Adversarial Probe (A2P). Specifically, we posit that the challenge of universal backdoor attacks detection lies in the fact that different backdoor attacks often exhibit diverse characteristics in trigger patterns (i.e., sizes and transparencies). Therefore, our A2P adopts a global-to-local probing framework, which adversarially probes images with adaptive regions/budgets to fit various backdoor triggers of different sizes/transparencies. Regarding the probing region, we propose the attention-guided region generation strategy that generates region proposals with different sizes/locations based on the attention of the target model, since trigger regions often manifest higher model activation. Considering the attack budget, we introduce the box-to-sparsity scheduling that iteratively increases the perturbation budget from box to sparse constraint, so that we could better activate different latent backdoors with different transparencies. Extensive experiments on multiple datasets (CIFAR-10, GTSRB, Tiny-ImageNet) demonstrate that our method outperforms state-of-the-art baselines by large margins (+12%).



## **3. LeNo: Adversarial Robust Salient Object Detection Networks with Learnable Noise**

cs.CV

8 pages, 6 figures, accepted by AAAI 2023

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2210.15392v2) [paper-pdf](http://arxiv.org/pdf/2210.15392v2)

**Authors**: He Wang, Lin Wan, He Tang

**Abstract**: Pixel-wise prediction with deep neural network has become an effective paradigm for salient object detection (SOD) and achieved remarkable performance. However, very few SOD models are robust against adversarial attacks which are visually imperceptible for human visual attention. The previous work robust saliency (ROSA) shuffles the pre-segmented superpixels and then refines the coarse saliency map by the densely connected conditional random field (CRF). Different from ROSA that relies on various pre- and post-processings, this paper proposes a light-weight Learnable Noise (LeNo) to defend adversarial attacks for SOD models. LeNo preserves accuracy of SOD models on both adversarial and clean images, as well as inference speed. In general, LeNo consists of a simple shallow noise and noise estimation that embedded in the encoder and decoder of arbitrary SOD networks respectively. Inspired by the center prior of human visual attention mechanism, we initialize the shallow noise with a cross-shaped gaussian distribution for better defense against adversarial attacks. Instead of adding additional network components for post-processing, the proposed noise estimation modifies only one channel of the decoder. With the deeply-supervised noise-decoupled training on state-of-the-art RGB and RGB-D SOD networks, LeNo outperforms previous works not only on adversarial images but also on clean images, which contributes stronger robustness for SOD. Our code is available at https://github.com/ssecv/LeNo.



## **4. Pre-trained Encoders in Self-Supervised Learning Improve Secure and Privacy-preserving Supervised Learning**

cs.CR

**SubmitDate**: 2022-12-06    [abs](http://arxiv.org/abs/2212.03334v1) [paper-pdf](http://arxiv.org/pdf/2212.03334v1)

**Authors**: Hongbin Liu, Wenjie Qu, Jinyuan Jia, Neil Zhenqiang Gong

**Abstract**: Classifiers in supervised learning have various security and privacy issues, e.g., 1) data poisoning attacks, backdoor attacks, and adversarial examples on the security side as well as 2) inference attacks and the right to be forgotten for the training data on the privacy side. Various secure and privacy-preserving supervised learning algorithms with formal guarantees have been proposed to address these issues. However, they suffer from various limitations such as accuracy loss, small certified security guarantees, and/or inefficiency. Self-supervised learning is an emerging technique to pre-train encoders using unlabeled data. Given a pre-trained encoder as a feature extractor, supervised learning can train a simple yet accurate classifier using a small amount of labeled training data. In this work, we perform the first systematic, principled measurement study to understand whether and when a pre-trained encoder can address the limitations of secure or privacy-preserving supervised learning algorithms. Our key findings are that a pre-trained encoder substantially improves 1) both accuracy under no attacks and certified security guarantees against data poisoning and backdoor attacks of state-of-the-art secure learning algorithms (i.e., bagging and KNN), 2) certified security guarantees of randomized smoothing against adversarial examples without sacrificing its accuracy under no attacks, 3) accuracy of differentially private classifiers, and 4) accuracy and/or efficiency of exact machine unlearning.



## **5. Robust Models are less Over-Confident**

cs.CV

accepted at NeurIPS 2022

**SubmitDate**: 2022-12-06    [abs](http://arxiv.org/abs/2210.05938v2) [paper-pdf](http://arxiv.org/pdf/2210.05938v2)

**Authors**: Julia Grabinski, Paul Gavrikov, Janis Keuper, Margret Keuper

**Abstract**: Despite the success of convolutional neural networks (CNNs) in many academic benchmarks for computer vision tasks, their application in the real-world is still facing fundamental challenges. One of these open problems is the inherent lack of robustness, unveiled by the striking effectiveness of adversarial attacks. Current attack methods are able to manipulate the network's prediction by adding specific but small amounts of noise to the input. In turn, adversarial training (AT) aims to achieve robustness against such attacks and ideally a better model generalization ability by including adversarial samples in the trainingset. However, an in-depth analysis of the resulting robust models beyond adversarial robustness is still pending. In this paper, we empirically analyze a variety of adversarially trained models that achieve high robust accuracies when facing state-of-the-art attacks and we show that AT has an interesting side-effect: it leads to models that are significantly less overconfident with their decisions, even on clean data than non-robust models. Further, our analysis of robust models shows that not only AT but also the model's building blocks (like activation functions and pooling) have a strong influence on the models' prediction confidences. Data & Project website: https://github.com/GeJulia/robustness_confidences_evaluation



## **6. On the tightness of linear relaxation based robustness certification methods**

cs.LG

**SubmitDate**: 2022-12-06    [abs](http://arxiv.org/abs/2210.00178v2) [paper-pdf](http://arxiv.org/pdf/2210.00178v2)

**Authors**: Cheng Tang

**Abstract**: There has been a rapid development and interest in adversarial training and defenses in the machine learning community in the recent years. One line of research focuses on improving the performance and efficiency of adversarial robustness certificates for neural networks \cite{gowal:19, wong_zico:18, raghunathan:18, WengTowardsFC:18, wong:scalable:18, singh:convex_barrier:19, Huang_etal:19, single-neuron-relax:20, Zhang2020TowardsSA}. While each providing a certification to lower (or upper) bound the true distortion under adversarial attacks via relaxation, less studied was the tightness of relaxation. In this paper, we analyze a family of linear outer approximation based certificate methods via a meta algorithm, IBP-Lin. The aforementioned works often lack quantitative analysis to answer questions such as how does the performance of the certificate method depend on the network configuration and the choice of approximation parameters. Under our framework, we make a first attempt at answering these questions, which reveals that the tightness of linear approximation based certification can depend heavily on the configuration of the trained networks.



## **7. CodeAttack: Code-based Adversarial Attacks for Pre-Trained Programming Language Models**

cs.CL

**SubmitDate**: 2022-12-06    [abs](http://arxiv.org/abs/2206.00052v2) [paper-pdf](http://arxiv.org/pdf/2206.00052v2)

**Authors**: Akshita Jha, Chandan K. Reddy

**Abstract**: Pre-trained programming language (PL) models (such as CodeT5, CodeBERT, GraphCodeBERT, etc.,) have the potential to automate software engineering tasks involving code understanding and code generation. However, these models operate in the natural channel of code, i.e., they are primarily concerned with the human understanding of the code. They are not robust to changes in the input and thus, are potentially susceptible to adversarial attacks in the natural channel. We propose, CodeAttack, a simple yet effective black-box attack model that uses code structure to generate effective, efficient, and imperceptible adversarial code samples and demonstrates the vulnerabilities of the state-of-the-art PL models to code-specific adversarial attacks. We evaluate the transferability of CodeAttack on several code-code (translation and repair) and code-NL (summarization) tasks across different programming languages. CodeAttack outperforms state-of-the-art adversarial NLP attack models to achieve the best overall drop in performance while being more efficient, imperceptible, consistent, and fluent. The code can be found at https://github.com/reddy-lab-code-research/CodeAttack.



## **8. StyleGAN as a Utility-Preserving Face De-identification Method**

cs.CV

**SubmitDate**: 2022-12-05    [abs](http://arxiv.org/abs/2212.02611v1) [paper-pdf](http://arxiv.org/pdf/2212.02611v1)

**Authors**: Seyyed Mohammad Sadegh Moosavi Khorzooghi, Shirin Nilizadeh

**Abstract**: Several face de-identification methods have been proposed to preserve users' privacy by obscuring their faces. These methods, however, can degrade the quality of photos, and they usually do not preserve the utility of faces, e.g., their age, gender, pose, and facial expression. Recently, advanced generative adversarial network models, such as StyleGAN, have been proposed, which generate realistic, high-quality imaginary faces. In this paper, we investigate the use of StyleGAN in generating de-identified faces through style mixing, where the styles or features of the target face and an auxiliary face get mixed to generate a de-identified face that carries the utilities of the target face. We examined this de-identification method with respect to preserving utility and privacy, by implementing several face detection, verification, and identification attacks. Through extensive experiments and also comparing with two state-of-the-art face de-identification methods, we show that StyleGAN preserves the quality and utility of the faces much better than the other approaches and also by choosing the style mixing levels correctly, it can preserve the privacy of the faces much better than other methods.



## **9. Enhancing Quantum Adversarial Robustness by Randomized Encodings**

quant-ph

**SubmitDate**: 2022-12-05    [abs](http://arxiv.org/abs/2212.02531v1) [paper-pdf](http://arxiv.org/pdf/2212.02531v1)

**Authors**: Weiyuan Gong, Dong Yuan, Weikang Li, Dong-Ling Deng

**Abstract**: The interplay between quantum physics and machine learning gives rise to the emergent frontier of quantum machine learning, where advanced quantum learning models may outperform their classical counterparts in solving certain challenging problems. However, quantum learning systems are vulnerable to adversarial attacks: adding tiny carefully-crafted perturbations on legitimate input samples can cause misclassifications. To address this issue, we propose a general scheme to protect quantum learning systems from adversarial attacks by randomly encoding the legitimate data samples through unitary or quantum error correction encoders. In particular, we rigorously prove that both global and local random unitary encoders lead to exponentially vanishing gradients (i.e. barren plateaus) for any variational quantum circuits that aim to add adversarial perturbations, independent of the input data and the inner structures of adversarial circuits and quantum classifiers. In addition, we prove a rigorous bound on the vulnerability of quantum classifiers under local unitary adversarial attacks. We show that random black-box quantum error correction encoders can protect quantum classifiers against local adversarial noises and their robustness increases as we concatenate error correction codes. To quantify the robustness enhancement, we adapt quantum differential privacy as a measure of the prediction stability for quantum classifiers. Our results establish versatile defense strategies for quantum classifiers against adversarial perturbations, which provide valuable guidance to enhance the reliability and security for both near-term and future quantum learning technologies.



## **10. Domain Constraints in Feature Space: Strengthening Robustness of Android Malware Detection against Realizable Adversarial Examples**

cs.LG

**SubmitDate**: 2022-12-05    [abs](http://arxiv.org/abs/2205.15128v2) [paper-pdf](http://arxiv.org/pdf/2205.15128v2)

**Authors**: Hamid Bostani, Zhuoran Liu, Zhengyu Zhao, Veelasha Moonsamy

**Abstract**: Strengthening the robustness of machine learning-based Android malware detectors in the real world requires incorporating realizable adversarial examples (RealAEs), i.e., AEs that satisfy the domain constraints of Android malware. However, existing work focuses on generating RealAEs in the problem space, which is known to be time-consuming and impractical for adversarial training. In this paper, we propose to generate RealAEs in the feature space, leading to a simpler and more efficient solution. Our approach is driven by a novel interpretation of Android malware properties in the feature space. More concretely, we extract feature-space domain constraints by learning meaningful feature dependencies from data and applying them by constructing a robust feature space. Our experiments on DREBIN, a well-known Android malware detector, demonstrate that our approach outperforms the state-of-the-art defense, Sec-SVM, against realistic gradient- and query-based attacks. Additionally, we demonstrate that generating feature-space RealAEs is faster than generating problem-space RealAEs, indicating its high applicability in adversarial training. We further validate the ability of our learned feature-space domain constraints in representing the Android malware properties by showing that (i) re-training detectors with our feature-space RealAEs largely improves model performance on similar problem-space RealAEs and (ii) using our feature-space domain constraints can help distinguish RealAEs from unrealizable AEs (unRealAEs).



## **11. Adversarial Attacks on Spiking Convolutional Neural Networks for Event-based Vision**

cs.CV

9 pages plus Supplementary Material. Accepted in Frontiers in  Neuroscience -- Neuromorphic Engineering

**SubmitDate**: 2022-12-05    [abs](http://arxiv.org/abs/2110.02929v3) [paper-pdf](http://arxiv.org/pdf/2110.02929v3)

**Authors**: Julian Büchel, Gregor Lenz, Yalun Hu, Sadique Sheik, Martino Sorbaro

**Abstract**: Event-based dynamic vision sensors provide very sparse output in the form of spikes, which makes them suitable for low-power applications. Convolutional spiking neural networks model such event-based data and develop their full energy-saving potential when deployed on asynchronous neuromorphic hardware. Event-based vision being a nascent field, the sensitivity of spiking neural networks to potentially malicious adversarial attacks has received little attention so far. We show how white-box adversarial attack algorithms can be adapted to the discrete and sparse nature of event-based visual data, and demonstrate smaller perturbation magnitudes at higher success rates than the current state-of-the-art algorithms. For the first time, we also verify the effectiveness of these perturbations directly on neuromorphic hardware. Finally, we discuss the properties of the resulting perturbations, the effect of adversarial training as a defense strategy, and future directions.



## **12. StyleFool: Fooling Video Classification Systems via Style Transfer**

cs.CV

18 pages, 9 figures. Accepted to S&P 2023

**SubmitDate**: 2022-12-05    [abs](http://arxiv.org/abs/2203.16000v2) [paper-pdf](http://arxiv.org/pdf/2203.16000v2)

**Authors**: Yuxin Cao, Xi Xiao, Ruoxi Sun, Derui Wang, Minhui Xue, Sheng Wen

**Abstract**: Video classification systems are vulnerable to adversarial attacks, which can create severe security problems in video verification. Current black-box attacks need a large number of queries to succeed, resulting in high computational overhead in the process of attack. On the other hand, attacks with restricted perturbations are ineffective against defenses such as denoising or adversarial training. In this paper, we focus on unrestricted perturbations and propose StyleFool, a black-box video adversarial attack via style transfer to fool the video classification system. StyleFool first utilizes color theme proximity to select the best style image, which helps avoid unnatural details in the stylized videos. Meanwhile, the target class confidence is additionally considered in targeted attacks to influence the output distribution of the classifier by moving the stylized video closer to or even across the decision boundary. A gradient-free method is then employed to further optimize the adversarial perturbations. We carry out extensive experiments to evaluate StyleFool on two standard datasets, UCF-101 and HMDB-51. The experimental results demonstrate that StyleFool outperforms the state-of-the-art adversarial attacks in terms of both the number of queries and the robustness against existing defenses. Moreover, 50% of the stylized videos in untargeted attacks do not need any query since they can already fool the video classification model. Furthermore, we evaluate the indistinguishability through a user study to show that the adversarial samples of StyleFool look imperceptible to human eyes, despite unrestricted perturbations.



## **13. FaceQAN: Face Image Quality Assessment Through Adversarial Noise Exploration**

cs.CV

The content of this paper was published in ICPR 2022

**SubmitDate**: 2022-12-05    [abs](http://arxiv.org/abs/2212.02127v1) [paper-pdf](http://arxiv.org/pdf/2212.02127v1)

**Authors**: Žiga Babnik, Peter Peer, Vitomir Štruc

**Abstract**: Recent state-of-the-art face recognition (FR) approaches have achieved impressive performance, yet unconstrained face recognition still represents an open problem. Face image quality assessment (FIQA) approaches aim to estimate the quality of the input samples that can help provide information on the confidence of the recognition decision and eventually lead to improved results in challenging scenarios. While much progress has been made in face image quality assessment in recent years, computing reliable quality scores for diverse facial images and FR models remains challenging. In this paper, we propose a novel approach to face image quality assessment, called FaceQAN, that is based on adversarial examples and relies on the analysis of adversarial noise which can be calculated with any FR model learned by using some form of gradient descent. As such, the proposed approach is the first to link image quality to adversarial attacks. Comprehensive (cross-model as well as model-specific) experiments are conducted with four benchmark datasets, i.e., LFW, CFP-FP, XQLFW and IJB-C, four FR models, i.e., CosFace, ArcFace, CurricularFace and ElasticFace, and in comparison to seven state-of-the-art FIQA methods to demonstrate the performance of FaceQAN. Experimental results show that FaceQAN achieves competitive results, while exhibiting several desirable characteristics.



## **14. Bayesian Learning with Information Gain Provably Bounds Risk for a Robust Adversarial Defense**

cs.LG

Published at ICML 2022

**SubmitDate**: 2022-12-05    [abs](http://arxiv.org/abs/2212.02003v1) [paper-pdf](http://arxiv.org/pdf/2212.02003v1)

**Authors**: Bao Gia Doan, Ehsan Abbasnejad, Javen Qinfeng Shi, Damith C. Ranasinghe

**Abstract**: We present a new algorithm to learn a deep neural network model robust against adversarial attacks. Previous algorithms demonstrate an adversarially trained Bayesian Neural Network (BNN) provides improved robustness. We recognize the adversarial learning approach for approximating the multi-modal posterior distribution of a Bayesian model can lead to mode collapse; consequently, the model's achievements in robustness and performance are sub-optimal. Instead, we first propose preventing mode collapse to better approximate the multi-modal posterior distribution. Second, based on the intuition that a robust model should ignore perturbations and only consider the informative content of the input, we conceptualize and formulate an information gain objective to measure and force the information learned from both benign and adversarial training instances to be similar. Importantly. we prove and demonstrate that minimizing the information gain objective allows the adversarial risk to approach the conventional empirical risk. We believe our efforts provide a step toward a basis for a principled method of adversarially training BNNs. Our model demonstrate significantly improved robustness--up to 20%--compared with adversarial training and Adv-BNN under PGD attacks with 0.035 distortion on both CIFAR-10 and STL-10 datasets.



## **15. Recognizing Object by Components with Human Prior Knowledge Enhances Adversarial Robustness of Deep Neural Networks**

cs.CV

Under review. Submitted to TPAMI on June 10, 2022. Major revision on  September 4, 2022

**SubmitDate**: 2022-12-04    [abs](http://arxiv.org/abs/2212.01806v1) [paper-pdf](http://arxiv.org/pdf/2212.01806v1)

**Authors**: Xiao Li, Ziqi Wang, Bo Zhang, Fuchun Sun, Xiaolin Hu

**Abstract**: Adversarial attacks can easily fool object recognition systems based on deep neural networks (DNNs). Although many defense methods have been proposed in recent years, most of them can still be adaptively evaded. One reason for the weak adversarial robustness may be that DNNs are only supervised by category labels and do not have part-based inductive bias like the recognition process of humans. Inspired by a well-known theory in cognitive psychology -- recognition-by-components, we propose a novel object recognition model ROCK (Recognizing Object by Components with human prior Knowledge). It first segments parts of objects from images, then scores part segmentation results with predefined human prior knowledge, and finally outputs prediction based on the scores. The first stage of ROCK corresponds to the process of decomposing objects into parts in human vision. The second stage corresponds to the decision process of the human brain. ROCK shows better robustness than classical recognition models across various attack settings. These results encourage researchers to rethink the rationality of currently widely-used DNN-based object recognition models and explore the potential of part-based models, once important but recently ignored, for improving robustness.



## **16. Imperceptible Adversarial Attack via Invertible Neural Networks**

cs.CV

**SubmitDate**: 2022-12-04    [abs](http://arxiv.org/abs/2211.15030v2) [paper-pdf](http://arxiv.org/pdf/2211.15030v2)

**Authors**: Zihan Chen, Ziyue Wang, Junjie Huang, Wentao Zhao, Xiao Liu, Dejian Guan

**Abstract**: Adding perturbations via utilizing auxiliary gradient information or discarding existing details of the benign images are two common approaches for generating adversarial examples. Though visual imperceptibility is the desired property of adversarial examples, conventional adversarial attacks still generate traceable adversarial perturbations. In this paper, we introduce a novel Adversarial Attack via Invertible Neural Networks (AdvINN) method to produce robust and imperceptible adversarial examples. Specifically, AdvINN fully takes advantage of the information preservation property of Invertible Neural Networks and thereby generates adversarial examples by simultaneously adding class-specific semantic information of the target class and dropping discriminant information of the original class. Extensive experiments on CIFAR-10, CIFAR-100, and ImageNet-1K demonstrate that the proposed AdvINN method can produce less imperceptible adversarial images than the state-of-the-art methods and AdvINN yields more robust adversarial examples with high confidence compared to other adversarial attacks.



## **17. LDL: A Defense for Label-Based Membership Inference Attacks**

cs.LG

to appear in ACM AsiaCCS 2023

**SubmitDate**: 2022-12-03    [abs](http://arxiv.org/abs/2212.01688v1) [paper-pdf](http://arxiv.org/pdf/2212.01688v1)

**Authors**: Arezoo Rajabi, Dinuka Sahabandu, Luyao Niu, Bhaskar Ramasubramanian, Radha Poovendran

**Abstract**: The data used to train deep neural network (DNN) models in applications such as healthcare and finance typically contain sensitive information. A DNN model may suffer from overfitting. Overfitted models have been shown to be susceptible to query-based attacks such as membership inference attacks (MIAs). MIAs aim to determine whether a sample belongs to the dataset used to train a classifier (members) or not (nonmembers). Recently, a new class of label based MIAs (LAB MIAs) was proposed, where an adversary was only required to have knowledge of predicted labels of samples. Developing a defense against an adversary carrying out a LAB MIA on DNN models that cannot be retrained remains an open problem.   We present LDL, a light weight defense against LAB MIAs. LDL works by constructing a high-dimensional sphere around queried samples such that the model decision is unchanged for (noisy) variants of the sample within the sphere. This sphere of label-invariance creates ambiguity and prevents a querying adversary from correctly determining whether a sample is a member or a nonmember. We analytically characterize the success rate of an adversary carrying out a LAB MIA when LDL is deployed, and show that the formulation is consistent with experimental observations. We evaluate LDL on seven datasets -- CIFAR-10, CIFAR-100, GTSRB, Face, Purchase, Location, and Texas -- with varying sizes of training data. All of these datasets have been used by SOTA LAB MIAs. Our experiments demonstrate that LDL reduces the success rate of an adversary carrying out a LAB MIA in each case. We empirically compare LDL with defenses against LAB MIAs that require retraining of DNN models, and show that LDL performs favorably despite not needing to retrain the DNNs.



## **18. Sparta: Spatially Attentive and Adversarially Robust Activation**

cs.LG

25 pages, 5 figures

**SubmitDate**: 2022-12-03    [abs](http://arxiv.org/abs/2105.08269v2) [paper-pdf](http://arxiv.org/pdf/2105.08269v2)

**Authors**: Qing Guo, Felix Juefei-Xu, Changqing Zhou, Wei Feng, Yang Liu, Song Wang

**Abstract**: Adversarial training (AT) is one of the most effective ways for improving the robustness of deep convolution neural networks (CNNs). Just like common network training, the effectiveness of AT relies on the design of basic network components. In this paper, we conduct an in-depth study on the role of the basic ReLU activation component in AT for robust CNNs. We find that the spatially-shared and input-independent properties of ReLU activation make CNNs less robust to white-box adversarial attacks with either standard or adversarial training. To address this problem, we extend ReLU to a novel Sparta activation function (Spatially attentive and Adversarially Robust Activation), which enables CNNs to achieve both higher robustness, i.e., lower error rate on adversarial examples, and higher accuracy, i.e., lower error rate on clean examples, than the existing state-of-the-art (SOTA) activation functions. We further study the relationship between Sparta and the SOTA activation functions, providing more insights about the advantages of our method. With comprehensive experiments, we also find that the proposed method exhibits superior cross-CNN and cross-dataset transferability. For the former, the adversarially trained Sparta function for one CNN (e.g., ResNet-18) can be fixed and directly used to train another adversarially robust CNN (e.g., ResNet-34). For the latter, the Sparta function trained on one dataset (e.g., CIFAR-10) can be employed to train adversarially robust CNNs on another dataset (e.g., SVHN). In both cases, Sparta leads to CNNs with higher robustness than the vanilla ReLU, verifying the flexibility and versatility of the proposed method.



## **19. Task and Model Agnostic Adversarial Attack on Graph Neural Networks**

cs.LG

To appear as a full paper in AAAI 2023

**SubmitDate**: 2022-12-03    [abs](http://arxiv.org/abs/2112.13267v2) [paper-pdf](http://arxiv.org/pdf/2112.13267v2)

**Authors**: Kartik Sharma, Samidha Verma, Sourav Medya, Sayan Ranu, Arnab Bhattacharya

**Abstract**: Adversarial attacks on Graph Neural Networks (GNNs) reveal their security vulnerabilities, limiting their adoption in safety-critical applications. However, existing attack strategies rely on the knowledge of either the GNN model being used or the predictive task being attacked. Is this knowledge necessary? For example, a graph may be used for multiple downstream tasks unknown to a practical attacker. It is thus important to test the vulnerability of GNNs to adversarial perturbations in a model and task agnostic setting. In this work, we study this problem and show that GNNs remain vulnerable even when the downstream task and model are unknown. The proposed algorithm, TANDIS (Targeted Attack via Neighborhood DIStortion) shows that distortion of node neighborhoods is effective in drastically compromising prediction performance. Although neighborhood distortion is an NP-hard problem, TANDIS designs an effective heuristic through a novel combination of Graph Isomorphism Network with deep Q-learning. Extensive experiments on real datasets and state-of-the-art models show that, on average, TANDIS is up to 50% more effective than state-of-the-art techniques, while being more than 1000 times faster.



## **20. RoS-KD: A Robust Stochastic Knowledge Distillation Approach for Noisy Medical Imaging**

cs.CV

Accepted in ICDM 2022

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2210.08388v2) [paper-pdf](http://arxiv.org/pdf/2210.08388v2)

**Authors**: Ajay Jaiswal, Kumar Ashutosh, Justin F Rousseau, Yifan Peng, Zhangyang Wang, Ying Ding

**Abstract**: AI-powered Medical Imaging has recently achieved enormous attention due to its ability to provide fast-paced healthcare diagnoses. However, it usually suffers from a lack of high-quality datasets due to high annotation cost, inter-observer variability, human annotator error, and errors in computer-generated labels. Deep learning models trained on noisy labelled datasets are sensitive to the noise type and lead to less generalization on the unseen samples. To address this challenge, we propose a Robust Stochastic Knowledge Distillation (RoS-KD) framework which mimics the notion of learning a topic from multiple sources to ensure deterrence in learning noisy information. More specifically, RoS-KD learns a smooth, well-informed, and robust student manifold by distilling knowledge from multiple teachers trained on overlapping subsets of training data. Our extensive experiments on popular medical imaging classification tasks (cardiopulmonary disease and lesion classification) using real-world datasets, show the performance benefit of RoS-KD, its ability to distill knowledge from many popular large networks (ResNet-50, DenseNet-121, MobileNet-V2) in a comparatively small network, and its robustness to adversarial attacks (PGD, FSGM). More specifically, RoS-KD achieves >2% and >4% improvement on F1-score for lesion classification and cardiopulmonary disease classification tasks, respectively, when the underlying student is ResNet-18 against recent competitive knowledge distillation baseline. Additionally, on cardiopulmonary disease classification task, RoS-KD outperforms most of the SOTA baselines by ~1% gain in AUC score.



## **21. Dual Graphs of Polyhedral Decompositions for the Detection of Adversarial Attacks**

cs.CV

978-1-6654-8045-1/22/\$31.00 \copyright{}2022 IEEE The 6th Workshop  on Graph Techniques for Adversarial Activity Analytics (GTA 2022)

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2211.13305v2) [paper-pdf](http://arxiv.org/pdf/2211.13305v2)

**Authors**: Huma Jamil, Yajing Liu, Christina M. Cole, Nathaniel Blanchard, Emily J. King, Michael Kirby, Christopher Peterson

**Abstract**: Previous work has shown that a neural network with the rectified linear unit (ReLU) activation function leads to a convex polyhedral decomposition of the input space. These decompositions can be represented by a dual graph with vertices corresponding to polyhedra and edges corresponding to polyhedra sharing a facet, which is a subgraph of a Hamming graph. This paper illustrates how one can utilize the dual graph to detect and analyze adversarial attacks in the context of digital images. When an image passes through a network containing ReLU nodes, the firing or non-firing at a node can be encoded as a bit ($1$ for ReLU activation, $0$ for ReLU non-activation). The sequence of all bit activations identifies the image with a bit vector, which identifies it with a polyhedron in the decomposition and, in turn, identifies it with a vertex in the dual graph. We identify ReLU bits that are discriminators between non-adversarial and adversarial images and examine how well collections of these discriminators can ensemble vote to build an adversarial image detector. Specifically, we examine the similarities and differences of ReLU bit vectors for adversarial images, and their non-adversarial counterparts, using a pre-trained ResNet-50 architecture. While this paper focuses on adversarial digital images, ResNet-50 architecture, and the ReLU activation function, our methods extend to other network architectures, activation functions, and types of datasets.



## **22. Finitely Repeated Adversarial Quantum Hypothesis Testing**

quant-ph

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2212.02314v1) [paper-pdf](http://arxiv.org/pdf/2212.02314v1)

**Authors**: Yinan Hu, Quanyan Zhu

**Abstract**: We formulate a passive quantum detector based on a quantum hypothesis testing framework under the setting of finite sample size. In particular, we exploit the fundamental limits of performance of the passive quantum detector asymptotically. Under the assumption that the attacker adopts separable optimal strategies, we derive that the worst-case average error bound converges to zero exponentially in terms of the number of repeated observations, which serves as a variation of quantum Sanov's theorem. We illustrate the general decaying results of miss rate numerically, depicting that the `naive' detector manages to achieve a miss rate and a false alarm rate both exponentially decaying to zero given infinitely many quantum states, although the miss rate decays to zero at a much slower rate than a quantum non-adversarial counterpart. Finally we adopt our formulations upon a case study of detection with quantum radars.



## **23. Diverse Generative Perturbations on Attention Space for Transferable Adversarial Attacks**

cs.CV

ICIP 2022 (Oral)

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2208.05650v2) [paper-pdf](http://arxiv.org/pdf/2208.05650v2)

**Authors**: Woo Jae Kim, Seunghoon Hong, Sung-Eui Yoon

**Abstract**: Adversarial attacks with improved transferability - the ability of an adversarial example crafted on a known model to also fool unknown models - have recently received much attention due to their practicality. Nevertheless, existing transferable attacks craft perturbations in a deterministic manner and often fail to fully explore the loss surface, thus falling into a poor local optimum and suffering from low transferability. To solve this problem, we propose Attentive-Diversity Attack (ADA), which disrupts diverse salient features in a stochastic manner to improve transferability. Primarily, we perturb the image attention to disrupt universal features shared by different models. Then, to effectively avoid poor local optima, we disrupt these features in a stochastic manner and explore the search space of transferable perturbations more exhaustively. More specifically, we use a generator to produce adversarial perturbations that each disturbs features in different ways depending on an input latent code. Extensive experimental evaluations demonstrate the effectiveness of our method, outperforming the transferability of state-of-the-art methods. Codes are available at https://github.com/wkim97/ADA.



## **24. Membership Inference Attacks Against Semantic Segmentation Models**

cs.CR

Submitted as conference paper to PETS 2023

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2212.01082v1) [paper-pdf](http://arxiv.org/pdf/2212.01082v1)

**Authors**: Tomas Chobola, Dmitrii Usynin, Georgios Kaissis

**Abstract**: Membership inference attacks aim to infer whether a data record has been used to train a target model by observing its predictions. In sensitive domains such as healthcare, this can constitute a severe privacy violation. In this work we attempt to address the existing knowledge gap by conducting an exhaustive study of membership inference attacks and defences in the domain of semantic image segmentation. Our findings indicate that for certain threat models, these learning settings can be considerably more vulnerable than the previously considered classification settings. We additionally investigate a threat model where a dishonest adversary can perform model poisoning to aid their inference and evaluate the effects that these adaptations have on the success of membership inference attacks. We quantitatively evaluate the attacks on a number of popular model architectures across a variety of semantic segmentation tasks, demonstrating that membership inference attacks in this domain can achieve a high success rate and defending against them may result in unfavourable privacy-utility trade-offs or increased computational costs.



## **25. Defending Black-box Skeleton-based Human Activity Classifiers**

cs.CV

Accepted in AAAI 2023

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2203.04713v6) [paper-pdf](http://arxiv.org/pdf/2203.04713v6)

**Authors**: He Wang, Yunfeng Diao, Zichang Tan, Guodong Guo

**Abstract**: Skeletal motions have been heavily replied upon for human activity recognition (HAR). Recently, a universal vulnerability of skeleton-based HAR has been identified across a variety of classifiers and data, calling for mitigation. To this end, we propose the first black-box defense method for skeleton-based HAR to our best knowledge. Our method is featured by full Bayesian treatments of the clean data, the adversaries and the classifier, leading to (1) a new Bayesian Energy-based formulation of robust discriminative classifiers, (2) a new adversary sampling scheme based on natural motion manifolds, and (3) a new post-train Bayesian strategy for black-box defense. We name our framework Bayesian Energy-based Adversarial Training or BEAT. BEAT is straightforward but elegant, which turns vulnerable black-box classifiers into robust ones without sacrificing accuracy. It demonstrates surprising and universal effectiveness across a wide range of skeletal HAR classifiers and datasets, under various attacks. Code is available at https://github.com/realcrane/RobustActionRecogniser.



## **26. AccEar: Accelerometer Acoustic Eavesdropping with Unconstrained Vocabulary**

cs.SD

2022 IEEE Symposium on Security and Privacy (SP)

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2212.01042v1) [paper-pdf](http://arxiv.org/pdf/2212.01042v1)

**Authors**: Pengfei Hu, Hui Zhuang, Panneer Selvam Santhalingamy, Riccardo Spolaor, Parth Pathaky, Guoming Zhang, Xiuzhen Cheng

**Abstract**: With the increasing popularity of voice-based applications, acoustic eavesdropping has become a serious threat to users' privacy. While on smartphones the access to microphones needs an explicit user permission, acoustic eavesdropping attacks can rely on motion sensors (such as accelerometer and gyroscope), which access is unrestricted. However, previous instances of such attacks can only recognize a limited set of pre-trained words or phrases. In this paper, we present AccEar, an accelerometerbased acoustic eavesdropping attack that can reconstruct any audio played on the smartphone's loudspeaker with unconstrained vocabulary. We show that an attacker can employ a conditional Generative Adversarial Network (cGAN) to reconstruct highfidelity audio from low-frequency accelerometer signals. The presented cGAN model learns to recreate high-frequency components of the user's voice from low-frequency accelerometer signals through spectrogram enhancement. We assess the feasibility and effectiveness of AccEar attack in a thorough set of experiments using audio from 16 public personalities. As shown by the results in both objective and subjective evaluations, AccEar successfully reconstructs user speeches from accelerometer signals in different scenarios including varying sampling rate, audio volume, device model, etc.



## **27. SecureSense: Defending Adversarial Attack for Secure Device-Free Human Activity Recognition**

cs.CR

The paper is accepted by IEEE Transactions on Mobile Computing

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2204.01560v2) [paper-pdf](http://arxiv.org/pdf/2204.01560v2)

**Authors**: Jianfei Yang, Han Zou, Lihua Xie

**Abstract**: Deep neural networks have empowered accurate device-free human activity recognition, which has wide applications. Deep models can extract robust features from various sensors and generalize well even in challenging situations such as data-insufficient cases. However, these systems could be vulnerable to input perturbations, i.e. adversarial attacks. We empirically demonstrate that both black-box Gaussian attacks and modern adversarial white-box attacks can render their accuracies to plummet. In this paper, we firstly point out that such phenomenon can bring severe safety hazards to device-free sensing systems, and then propose a novel learning framework, SecureSense, to defend common attacks. SecureSense aims to achieve consistent predictions regardless of whether there exists an attack on its input or not, alleviating the negative effect of distribution perturbation caused by adversarial attacks. Extensive experiments demonstrate that our proposed method can significantly enhance the model robustness of existing deep models, overcoming possible attacks. The results validate that our method works well on wireless human activity recognition and person identification systems. To the best of our knowledge, this is the first work to investigate adversarial attacks and further develop a novel defense framework for wireless human activity recognition in mobile computing research.



## **28. RamBoAttack: A Robust Query Efficient Deep Neural Network Decision Exploit**

cs.LG

Published in Network and Distributed System Security (NDSS) Symposium  2022

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2112.05282v2) [paper-pdf](http://arxiv.org/pdf/2112.05282v2)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: Machine learning models are critically susceptible to evasion attacks from adversarial examples. Generally, adversarial examples, modified inputs deceptively similar to the original input, are constructed under whitebox settings by adversaries with full access to the model. However, recent attacks have shown a remarkable reduction in query numbers to craft adversarial examples using blackbox attacks. Particularly, alarming is the ability to exploit the classification decision from the access interface of a trained model provided by a growing number of Machine Learning as a Service providers including Google, Microsoft, IBM and used by a plethora of applications incorporating these models. The ability of an adversary to exploit only the predicted label from a model to craft adversarial examples is distinguished as a decision-based attack. In our study, we first deep dive into recent state-of-the-art decision-based attacks in ICLR and SP to highlight the costly nature of discovering low distortion adversarial employing gradient estimation methods. We develop a robust query efficient attack capable of avoiding entrapment in a local minimum and misdirection from noisy gradients seen in gradient estimation methods. The attack method we propose, RamBoAttack, exploits the notion of Randomized Block Coordinate Descent to explore the hidden classifier manifold, targeting perturbations to manipulate only localized input features to address the issues of gradient estimation methods. Importantly, the RamBoAttack is more robust to the different sample inputs available to an adversary and the targeted class. Overall, for a given target class, RamBoAttack is demonstrated to be more robust at achieving a lower distortion within a given query budget. We curate our extensive results using the large-scale high-resolution ImageNet dataset and open-source our attack, test samples and artifacts on GitHub.



## **29. GARNET: Reduced-Rank Topology Learning for Robust and Scalable Graph Neural Networks**

cs.LG

Published as a conference paper at LoG 2022

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2201.12741v4) [paper-pdf](http://arxiv.org/pdf/2201.12741v4)

**Authors**: Chenhui Deng, Xiuyu Li, Zhuo Feng, Zhiru Zhang

**Abstract**: Graph neural networks (GNNs) have been increasingly deployed in various applications that involve learning on non-Euclidean data. However, recent studies show that GNNs are vulnerable to graph adversarial attacks. Although there are several defense methods to improve GNN robustness by eliminating adversarial components, they may also impair the underlying clean graph structure that contributes to GNN training. In addition, few of those defense models can scale to large graphs due to their high computational complexity and memory usage. In this paper, we propose GARNET, a scalable spectral method to boost the adversarial robustness of GNN models. GARNET first leverages weighted spectral embedding to construct a base graph, which is not only resistant to adversarial attacks but also contains critical (clean) graph structure for GNN training. Next, GARNET further refines the base graph by pruning additional uncritical edges based on probabilistic graphical model. GARNET has been evaluated on various datasets, including a large graph with millions of nodes. Our extensive experiment results show that GARNET achieves adversarial accuracy improvement and runtime speedup over state-of-the-art GNN (defense) models by up to 13.27% and 14.7x, respectively.



## **30. Pareto Regret Analyses in Multi-objective Multi-armed Bandit**

cs.LG

18 pages

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2212.00884v1) [paper-pdf](http://arxiv.org/pdf/2212.00884v1)

**Authors**: Mengfan Xu, Diego Klabjan

**Abstract**: We study Pareto optimality in multi-objective multi-armed bandit by providing a formulation of adversarial multi-objective multi-armed bandit and properly defining its Pareto regrets that can be generalized to stochastic settings as well. The regrets do not rely on any scalarization functions and reflect Pareto optimality compared to scalarized regrets. We also present new algorithms assuming both with and without prior information of the multi-objective multi-armed bandit setting. The algorithms are shown optimal in adversarial settings and nearly optimal in stochastic settings simultaneously by our established upper bounds and lower bounds on Pareto regrets. Moreover, the lower bound analyses show that the new regrets are consistent with the existing Pareto regret for stochastic settings and extend an adversarial attack mechanism from bandit to the multi-objective one.



## **31. Adversarial Analysis of the Differentially-Private Federated Learning in Cyber-Physical Critical Infrastructures**

cs.CR

16 pages, 9 figures, 5 tables. This work has been submitted to IEEE  for possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2204.02654v2) [paper-pdf](http://arxiv.org/pdf/2204.02654v2)

**Authors**: Md Tamjid Hossain, Shahriar Badsha, Hung La, Haoting Shen, Shafkat Islam, Ibrahim Khalil, Xun Yi

**Abstract**: Federated Learning (FL) has become increasingly popular to perform data-driven analysis in cyber-physical critical infrastructures. Since the FL process may involve the client's confidential information, Differential Privacy (DP) has been proposed lately to secure it from adversarial inference. However, we find that while DP greatly alleviates the privacy concerns, the additional DP-noise opens a new threat for model poisoning in FL. Nonetheless, very little effort has been made in the literature to investigate this adversarial exploitation of the DP-noise. To overcome this gap, in this paper, we present a novel adaptive model poisoning technique {\alpha}-MPELM} through which an attacker can exploit the additional DP-noise to evade the state-of-the-art anomaly detection techniques and prevent optimal convergence of the FL model. We evaluate our proposed attack on the state-of-the-art anomaly detection approaches in terms of detection accuracy and validation loss. The main significance of our proposed {\alpha}-MPELM attack is that it reduces the state-of-the-art anomaly detection accuracy by 6.8% for norm detection, 12.6% for accuracy detection, and 13.8% for mix detection. Furthermore, we propose a Reinforcement Learning-based DP level selection process to defend {\alpha}-MPELM attack. The experimental results confirm that our defense mechanism converges to an optimal privacy policy without human maneuver.



## **32. Purifier: Defending Data Inference Attacks via Transforming Confidence Scores**

cs.LG

accepted by AAAI 2023

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2212.00612v1) [paper-pdf](http://arxiv.org/pdf/2212.00612v1)

**Authors**: Ziqi Yang, Lijin Wang, Da Yang, Jie Wan, Ziming Zhao, Ee-Chien Chang, Fan Zhang, Kui Ren

**Abstract**: Neural networks are susceptible to data inference attacks such as the membership inference attack, the adversarial model inversion attack and the attribute inference attack, where the attacker could infer useful information such as the membership, the reconstruction or the sensitive attributes of a data sample from the confidence scores predicted by the target classifier. In this paper, we propose a method, namely PURIFIER, to defend against membership inference attacks. It transforms the confidence score vectors predicted by the target classifier and makes purified confidence scores indistinguishable in individual shape, statistical distribution and prediction label between members and non-members. The experimental results show that PURIFIER helps defend membership inference attacks with high effectiveness and efficiency, outperforming previous defense methods, and also incurs negligible utility loss. Besides, our further experiments show that PURIFIER is also effective in defending adversarial model inversion attacks and attribute inference attacks. For example, the inversion error is raised about 4+ times on the Facescrub530 classifier, and the attribute inference accuracy drops significantly when PURIFIER is deployed in our experiment.



## **33. PointCA: Evaluating the Robustness of 3D Point Cloud Completion Models Against Adversarial Examples**

cs.CV

Accepted by the 37th AAAI Conference on Artificial Intelligence  (AAAI-23)

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2211.12294v2) [paper-pdf](http://arxiv.org/pdf/2211.12294v2)

**Authors**: Shengshan Hu, Junwei Zhang, Wei Liu, Junhui Hou, Minghui Li, Leo Yu Zhang, Hai Jin, Lichao Sun

**Abstract**: Point cloud completion, as the upstream procedure of 3D recognition and segmentation, has become an essential part of many tasks such as navigation and scene understanding. While various point cloud completion models have demonstrated their powerful capabilities, their robustness against adversarial attacks, which have been proven to be fatally malicious towards deep neural networks, remains unknown. In addition, existing attack approaches towards point cloud classifiers cannot be applied to the completion models due to different output forms and attack purposes. In order to evaluate the robustness of the completion models, we propose PointCA, the first adversarial attack against 3D point cloud completion models. PointCA can generate adversarial point clouds that maintain high similarity with the original ones, while being completed as another object with totally different semantic information. Specifically, we minimize the representation discrepancy between the adversarial example and the target point set to jointly explore the adversarial point clouds in the geometry space and the feature space. Furthermore, to launch a stealthier attack, we innovatively employ the neighbourhood density information to tailor the perturbation constraint, leading to geometry-aware and distribution-adaptive modifications for each point. Extensive experiments against different premier point cloud completion networks show that PointCA can cause a performance degradation from 77.9% to 16.7%, with the structure chamfer distance kept below 0.01. We conclude that existing completion models are severely vulnerable to adversarial examples, and state-of-the-art defenses for point cloud classification will be partially invalid when applied to incomplete and uneven point cloud data.



## **34. All You Need Is Hashing: Defending Against Data Reconstruction Attack in Vertical Federated Learning**

cs.CR

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2212.00325v1) [paper-pdf](http://arxiv.org/pdf/2212.00325v1)

**Authors**: Pengyu Qiu, Xuhong Zhang, Shouling Ji, Yuwen Pu, Ting Wang

**Abstract**: Vertical federated learning is a trending solution for multi-party collaboration in training machine learning models. Industrial frameworks adopt secure multi-party computation methods such as homomorphic encryption to guarantee data security and privacy. However, a line of work has revealed that there are still leakage risks in VFL. The leakage is caused by the correlation between the intermediate representations and the raw data. Due to the powerful approximation ability of deep neural networks, an adversary can capture the correlation precisely and reconstruct the data. To deal with the threat of the data reconstruction attack, we propose a hashing-based VFL framework, called \textit{HashVFL}, to cut off the reversibility directly. The one-way nature of hashing allows our framework to block all attempts to recover data from hash codes. However, integrating hashing also brings some challenges, e.g., the loss of information. This paper proposes and addresses three challenges to integrating hashing: learnability, bit balance, and consistency. Experimental results demonstrate \textit{HashVFL}'s efficiency in keeping the main task's performance and defending against data reconstruction attacks. Furthermore, we also analyze its potential value in detecting abnormal inputs. In addition, we conduct extensive experiments to prove \textit{HashVFL}'s generalization in various settings. In summary, \textit{HashVFL} provides a new perspective on protecting multi-party's data security and privacy in VFL. We hope our study can attract more researchers to expand the application domains of \textit{HashVFL}.



## **35. Overcoming the Convex Relaxation Barrier for Neural Network Verification via Nonconvex Low-Rank Semidefinite Relaxations**

cs.LG

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2211.17244v1) [paper-pdf](http://arxiv.org/pdf/2211.17244v1)

**Authors**: Hong-Ming Chiu, Richard Y. Zhang

**Abstract**: To rigorously certify the robustness of neural networks to adversarial perturbations, most state-of-the-art techniques rely on a triangle-shaped linear programming (LP) relaxation of the ReLU activation. While the LP relaxation is exact for a single neuron, recent results suggest that it faces an inherent "convex relaxation barrier" as additional activations are added, and as the attack budget is increased. In this paper, we propose a nonconvex relaxation for the ReLU relaxation, based on a low-rank restriction of a semidefinite programming (SDP) relaxation. We show that the nonconvex relaxation has a similar complexity to the LP relaxation, but enjoys improved tightness that is comparable to the much more expensive SDP relaxation. Despite nonconvexity, we prove that the verification problem satisfies constraint qualification, and therefore a Riemannian staircase approach is guaranteed to compute a near-globally optimal solution in polynomial time. Our experiments provide evidence that our nonconvex relaxation almost completely overcome the "convex relaxation barrier" faced by the LP relaxation.



## **36. Differentially Private ADMM-Based Distributed Discrete Optimal Transport for Resource Allocation**

cs.SI

6 pages, 4 images, 1 algorithm, IEEE GLOBECOMM 2022

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2211.17070v1) [paper-pdf](http://arxiv.org/pdf/2211.17070v1)

**Authors**: Jason Hughes, Juntao Chen

**Abstract**: Optimal transport (OT) is a framework that can guide the design of efficient resource allocation strategies in a network of multiple sources and targets. To ease the computational complexity of large-scale transport design, we first develop a distributed algorithm based on the alternating direction method of multipliers (ADMM). However, such a distributed algorithm is vulnerable to sensitive information leakage when an attacker intercepts the transport decisions communicated between nodes during the distributed ADMM updates. To this end, we propose a privacy-preserving distributed mechanism based on output variable perturbation by adding appropriate randomness to each node's decision before it is shared with other corresponding nodes at each update instance. We show that the developed scheme is differentially private, which prevents the adversary from inferring the node's confidential information even knowing the transport decisions. Finally, we corroborate the effectiveness of the devised algorithm through case studies.



## **37. A Systematic Evaluation of Node Embedding Robustness**

cs.LG

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2209.08064v3) [paper-pdf](http://arxiv.org/pdf/2209.08064v3)

**Authors**: Alexandru Mara, Jefrey Lijffijt, Stephan Günnemann, Tijl De Bie

**Abstract**: Node embedding methods map network nodes to low dimensional vectors that can be subsequently used in a variety of downstream prediction tasks. The popularity of these methods has grown significantly in recent years, yet, their robustness to perturbations of the input data is still poorly understood. In this paper, we assess the empirical robustness of node embedding models to random and adversarial poisoning attacks. Our systematic evaluation covers representative embedding methods based on Skip-Gram, matrix factorization, and deep neural networks. We compare edge addition, deletion and rewiring attacks computed using network properties as well as node labels. We also investigate the performance of popular node classification attack baselines that assume full knowledge of the node labels. We report qualitative results via embedding visualization and quantitative results in terms of downstream node classification and network reconstruction performances. We find that node classification results are impacted more than network reconstruction ones, that degree-based and label-based attacks are on average the most damaging and that label heterophily can strongly influence attack performance.



## **38. Adaptive adversarial training method for improving multi-scale GAN based on generalization bound theory**

cs.CV

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2211.16791v1) [paper-pdf](http://arxiv.org/pdf/2211.16791v1)

**Authors**: Jing Tang, Bo Tao, Zeyu Gong, Zhouping Yin

**Abstract**: In recent years, multi-scale generative adversarial networks (GANs) have been proposed to build generalized image processing models based on single sample. Constraining on the sample size, multi-scale GANs have much difficulty converging to the global optimum, which ultimately leads to limitations in their capabilities. In this paper, we pioneered the introduction of PAC-Bayes generalized bound theory into the training analysis of specific models under different adversarial training methods, which can obtain a non-vacuous upper bound on the generalization error for the specified multi-scale GAN structure. Based on the drastic changes we found of the generalization error bound under different adversarial attacks and different training states, we proposed an adaptive training method which can greatly improve the image manipulation ability of multi-scale GANs. The final experimental results show that our adaptive training method in this paper has greatly contributed to the improvement of the quality of the images generated by multi-scale GANs on several image manipulation tasks. In particular, for the image super-resolution restoration task, the multi-scale GAN model trained by the proposed method achieves a 100% reduction in natural image quality evaluator (NIQE) and a 60% reduction in root mean squared error (RMSE), which is better than many models trained on large-scale datasets.



## **39. Sludge for Good: Slowing and Imposing Costs on Cyber Attackers**

cs.CR

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16626v1) [paper-pdf](http://arxiv.org/pdf/2211.16626v1)

**Authors**: Josiah Dykstra, Kelly Shortridge, Jamie Met, Douglas Hough

**Abstract**: Choice architecture describes the design by which choices are presented to people. Nudges are an aspect intended to make "good" outcomes easy, such as using password meters to encourage strong passwords. Sludge, on the contrary, is friction that raises the transaction cost and is often seen as a negative to users. Turning this concept around, we propose applying sludge for positive cybersecurity outcomes by using it offensively to consume attackers' time and other resources.   To date, most cyber defenses have been designed to be optimally strong and effective and prohibit or eliminate attackers as quickly as possible. Our complimentary approach is to also deploy defenses that seek to maximize the consumption of the attackers' time and other resources while causing as little damage as possible to the victim. This is consistent with zero trust and similar mindsets which assume breach. The Sludge Strategy introduces cost-imposing cyber defense by strategically deploying friction for attackers before, during, and after an attack using deception and authentic design features. We present the characteristics of effective sludge, and show a continuum from light to heavy sludge. We describe the quantitative and qualitative costs to attackers and offer practical considerations for deploying sludge in practice. Finally, we examine real-world examples of U.S. government operations to frustrate and impose cost on cyber adversaries.



## **40. Synthesizing Attack-Aware Control and Active Sensing Strategies under Reactive Sensor Attacks**

math.OC

7 pages, 3 figure, 1 table, 1 algorithm

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2204.01584v2) [paper-pdf](http://arxiv.org/pdf/2204.01584v2)

**Authors**: Sumukha Udupa, Abhishek N. Kulkarni, Shuo Han, Nandi O. Leslie, Charles A. Kamhoua, Jie Fu

**Abstract**: We consider the probabilistic planning problem for a defender (P1) who can jointly query the sensors and take control actions to reach a set of goal states while being aware of possible sensor attacks by an adversary (P2) who has perfect observations. To synthesize a provably-correct, attack-aware joint control and active sensing strategy for P1, we construct a stochastic game on graph with augmented states that include the actual game state (known only to the attacker), the belief of the defender about the game state (constructed by the attacker based on his knowledge of defender's observations). We present an algorithm to compute a belief-based, randomized strategy for P1 to ensure satisfying the reachability objective with probability one, under the worst-case sensor attack carried out by an informed P2. We prove the correctness of the algorithm and illustrate using an example.



## **41. Improving Adversarial Robustness with Self-Paced Hard-Class Pair Reweighting**

cs.CV

AAAI-23

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2210.15068v2) [paper-pdf](http://arxiv.org/pdf/2210.15068v2)

**Authors**: Pengyue Hou, Jie Han, Xingyu Li

**Abstract**: Deep Neural Networks are vulnerable to adversarial attacks. Among many defense strategies, adversarial training with untargeted attacks is one of the most effective methods. Theoretically, adversarial perturbation in untargeted attacks can be added along arbitrary directions and the predicted labels of untargeted attacks should be unpredictable. However, we find that the naturally imbalanced inter-class semantic similarity makes those hard-class pairs become virtual targets of each other. This study investigates the impact of such closely-coupled classes on adversarial attacks and develops a self-paced reweighting strategy in adversarial training accordingly. Specifically, we propose to upweight hard-class pair losses in model optimization, which prompts learning discriminative features from hard classes. We further incorporate a term to quantify hard-class pair consistency in adversarial training, which greatly boosts model robustness. Extensive experiments show that the proposed adversarial training method achieves superior robustness performance over state-of-the-art defenses against a wide range of adversarial attacks.



## **42. Resilient Risk based Adaptive Authentication and Authorization (RAD-AA) Framework**

cs.CR

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2208.02592v3) [paper-pdf](http://arxiv.org/pdf/2208.02592v3)

**Authors**: Jaimandeep Singh, Chintan Patel, Naveen Kumar Chaudhary

**Abstract**: In recent cyber attacks, credential theft has emerged as one of the primary vectors of gaining entry into the system. Once attacker(s) have a foothold in the system, they use various techniques including token manipulation to elevate the privileges and access protected resources. This makes authentication and token based authorization a critical component for a secure and resilient cyber system. In this paper we discuss the design considerations for such a secure and resilient authentication and authorization framework capable of self-adapting based on the risk scores and trust profiles. We compare this design with the existing standards such as OAuth 2.0, OpenID Connect and SAML 2.0. We then study popular threat models such as STRIDE and PASTA and summarize the resilience of the proposed architecture against common and relevant threat vectors. We call this framework as Resilient Risk based Adaptive Authentication and Authorization (RAD-AA). The proposed framework excessively increases the cost for an adversary to launch and sustain any cyber attack and provides much-needed strength to critical infrastructure. We also discuss the machine learning (ML) approach for the adaptive engine to accurately classify transactions and arrive at risk scores.



## **43. Ada3Diff: Defending against 3D Adversarial Point Clouds via Adaptive Diffusion**

cs.CV

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16247v1) [paper-pdf](http://arxiv.org/pdf/2211.16247v1)

**Authors**: Kui Zhang, Hang Zhou, Jie Zhang, Qidong Huang, Weiming Zhang, Nenghai Yu

**Abstract**: Deep 3D point cloud models are sensitive to adversarial attacks, which poses threats to safety-critical applications such as autonomous driving. Robust training and defend-by-denoise are typical strategies for defending adversarial perturbations, including adversarial training and statistical filtering, respectively. However, they either induce massive computational overhead or rely heavily upon specified noise priors, limiting generalized robustness against attacks of all kinds. This paper introduces a new defense mechanism based on denoising diffusion models that can adaptively remove diverse noises with a tailored intensity estimator. Specifically, we first estimate adversarial distortions by calculating the distance of the points to their neighborhood best-fit plane. Depending on the distortion degree, we choose specific diffusion time steps for the input point cloud and perform the forward diffusion to disrupt potential adversarial shifts. Then we conduct the reverse denoising process to restore the disrupted point cloud back to a clean distribution. This approach enables effective defense against adaptive attacks with varying noise budgets, achieving accentuated robustness of existing 3D deep recognition models.



## **44. Defending Adversarial Attacks on Deep Learning Based Power Allocation in Massive MIMO Using Denoising Autoencoders**

eess.SP

This work is currently under review for publication

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.15365v2) [paper-pdf](http://arxiv.org/pdf/2211.15365v2)

**Authors**: Rajeev Sahay, Minjun Zhang, David J. Love, Christopher G. Brinton

**Abstract**: Recent work has advocated for the use of deep learning to perform power allocation in the downlink of massive MIMO (maMIMO) networks. Yet, such deep learning models are vulnerable to adversarial attacks. In the context of maMIMO power allocation, adversarial attacks refer to the injection of subtle perturbations into the deep learning model's input, during inference (i.e., the adversarial perturbation is injected into inputs during deployment after the model has been trained) that are specifically crafted to force the trained regression model to output an infeasible power allocation solution. In this work, we develop an autoencoder-based mitigation technique, which allows deep learning-based power allocation models to operate in the presence of adversaries without requiring retraining. Specifically, we develop a denoising autoencoder (DAE), which learns a mapping between potentially perturbed data and its corresponding unperturbed input. We test our defense across multiple attacks and in multiple threat models and demonstrate its ability to (i) mitigate the effects of adversarial attacks on power allocation networks using two common precoding schemes, (ii) outperform previously proposed benchmarks for mitigating regression-based adversarial attacks on maMIMO networks, (iii) retain accurate performance in the absence of an attack, and (iv) operate with low computational overhead.



## **45. Quantization-aware Interval Bound Propagation for Training Certifiably Robust Quantized Neural Networks**

cs.LG

Accepted at AAAI 2023

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16187v1) [paper-pdf](http://arxiv.org/pdf/2211.16187v1)

**Authors**: Mathias Lechner, Đorđe Žikelić, Krishnendu Chatterjee, Thomas A. Henzinger, Daniela Rus

**Abstract**: We study the problem of training and certifying adversarially robust quantized neural networks (QNNs). Quantization is a technique for making neural networks more efficient by running them using low-bit integer arithmetic and is therefore commonly adopted in industry. Recent work has shown that floating-point neural networks that have been verified to be robust can become vulnerable to adversarial attacks after quantization, and certification of the quantized representation is necessary to guarantee robustness. In this work, we present quantization-aware interval bound propagation (QA-IBP), a novel method for training robust QNNs. Inspired by advances in robust learning of non-quantized networks, our training algorithm computes the gradient of an abstract representation of the actual network. Unlike existing approaches, our method can handle the discrete semantics of QNNs. Based on QA-IBP, we also develop a complete verification procedure for verifying the adversarial robustness of QNNs, which is guaranteed to terminate and produce a correct answer. Compared to existing approaches, the key advantage of our verification procedure is that it runs entirely on GPU or other accelerator devices. We demonstrate experimentally that our approach significantly outperforms existing methods and establish the new state-of-the-art for training and certifying the robustness of QNNs.



## **46. Understanding and Enhancing Robustness of Concept-based Models**

cs.LG

Accepted at AAAI 2023. Extended Version

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16080v1) [paper-pdf](http://arxiv.org/pdf/2211.16080v1)

**Authors**: Sanchit Sinha, Mengdi Huai, Jianhui Sun, Aidong Zhang

**Abstract**: Rising usage of deep neural networks to perform decision making in critical applications like medical diagnosis and financial analysis have raised concerns regarding their reliability and trustworthiness. As automated systems become more mainstream, it is important their decisions be transparent, reliable and understandable by humans for better trust and confidence. To this effect, concept-based models such as Concept Bottleneck Models (CBMs) and Self-Explaining Neural Networks (SENN) have been proposed which constrain the latent space of a model to represent high level concepts easily understood by domain experts in the field. Although concept-based models promise a good approach to both increasing explainability and reliability, it is yet to be shown if they demonstrate robustness and output consistent concepts under systematic perturbations to their inputs. To better understand performance of concept-based models on curated malicious samples, in this paper, we aim to study their robustness to adversarial perturbations, which are also known as the imperceptible changes to the input data that are crafted by an attacker to fool a well-learned concept-based model. Specifically, we first propose and analyze different malicious attacks to evaluate the security vulnerability of concept based models. Subsequently, we propose a potential general adversarial training-based defense mechanism to increase robustness of these systems to the proposed malicious attacks. Extensive experiments on one synthetic and two real-world datasets demonstrate the effectiveness of the proposed attacks and the defense approach.



## **47. Model Extraction Attack against Self-supervised Speech Models**

cs.SD

Submitted to ICASSP 2023

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16044v1) [paper-pdf](http://arxiv.org/pdf/2211.16044v1)

**Authors**: Tsu-Yuan Hsu, Chen-An Li, Tung-Yu Wu, Hung-yi Lee

**Abstract**: Self-supervised learning (SSL) speech models generate meaningful representations of given clips and achieve incredible performance across various downstream tasks. Model extraction attack (MEA) often refers to an adversary stealing the functionality of the victim model with only query access. In this work, we study the MEA problem against SSL speech model with a small number of queries. We propose a two-stage framework to extract the model. In the first stage, SSL is conducted on the large-scale unlabeled corpus to pre-train a small speech model. Secondly, we actively sample a small portion of clips from the unlabeled corpus and query the target model with these clips to acquire their representations as labels for the small model's second-stage training. Experiment results show that our sampling methods can effectively extract the target model without knowing any information about its model architecture.



## **48. AdvMask: A Sparse Adversarial Attack Based Data Augmentation Method for Image Classification**

cs.CV

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16040v1) [paper-pdf](http://arxiv.org/pdf/2211.16040v1)

**Authors**: Suorong Yang, Jinqiao Li, Jian Zhao, Furao Shen

**Abstract**: Data augmentation is a widely used technique for enhancing the generalization ability of convolutional neural networks (CNNs) in image classification tasks. Occlusion is a critical factor that affects on the generalization ability of image classification models. In order to generate new samples, existing data augmentation methods based on information deletion simulate occluded samples by randomly removing some areas in the images. However, those methods cannot delete areas of the images according to their structural features of the images. To solve those problems, we propose a novel data augmentation method, AdvMask, for image classification tasks. Instead of randomly removing areas in the images, AdvMask obtains the key points that have the greatest influence on the classification results via an end-to-end sparse adversarial attack module. Therefore, we can find the most sensitive points of the classification results without considering the diversity of various image appearance and shapes of the object of interest. In addition, a data augmentation module is employed to generate structured masks based on the key points, thus forcing the CNN classification models to seek other relevant content when the most discriminative content is hidden. AdvMask can effectively improve the performance of classification models in the testing process. The experimental results on various datasets and CNN models verify that the proposed method outperforms other previous data augmentation methods in image classification tasks.



## **49. Interpretations Cannot Be Trusted: Stealthy and Effective Adversarial Perturbations against Interpretable Deep Learning**

cs.CR

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.15926v1) [paper-pdf](http://arxiv.org/pdf/2211.15926v1)

**Authors**: Eldor Abdukhamidov, Mohammed Abuhamad, Simon S. Woo, Eric Chan-Tin, Tamer Abuhmed

**Abstract**: Deep learning methods have gained increased attention in various applications due to their outstanding performance. For exploring how this high performance relates to the proper use of data artifacts and the accurate problem formulation of a given task, interpretation models have become a crucial component in developing deep learning-based systems. Interpretation models enable the understanding of the inner workings of deep learning models and offer a sense of security in detecting the misuse of artifacts in the input data. Similar to prediction models, interpretation models are also susceptible to adversarial inputs. This work introduces two attacks, AdvEdge and AdvEdge$^{+}$, that deceive both the target deep learning model and the coupled interpretation model. We assess the effectiveness of proposed attacks against two deep learning model architectures coupled with four interpretation models that represent different categories of interpretation models. Our experiments include the attack implementation using various attack frameworks. We also explore the potential countermeasures against such attacks. Our analysis shows the effectiveness of our attacks in terms of deceiving the deep learning models and their interpreters, and highlights insights to improve and circumvent the attacks.



## **50. Training Time Adversarial Attack Aiming the Vulnerability of Continual Learning**

cs.LG

Accepted at NeurIPS 2022 ML Safety Workshop

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.15875v1) [paper-pdf](http://arxiv.org/pdf/2211.15875v1)

**Authors**: Gyojin Han, Jaehyun Choi, Hyeong Gwon Hong, Junmo Kim

**Abstract**: Generally, regularization-based continual learning models limit access to the previous task data to imitate the real-world setting which has memory and privacy issues. However, this introduces a problem in these models by not being able to track the performance on each task. In other words, current continual learning methods are vulnerable to attacks done on the previous task. We demonstrate the vulnerability of regularization-based continual learning methods by presenting simple task-specific training time adversarial attack that can be used in the learning process of a new task. Training data generated by the proposed attack causes performance degradation on a specific task targeted by the attacker. Experiment results justify the vulnerability proposed in this paper and demonstrate the importance of developing continual learning models that are robust to adversarial attack.



