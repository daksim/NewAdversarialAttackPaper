# Latest Adversarial Attack Papers
**update at 2022-03-05 06:31:55**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Assessing the Robustness of Visual Question Answering Models**

cs.CV

24 pages, 13 figures, International Journal of Computer Vision (IJCV)  [under review]. arXiv admin note: substantial text overlap with  arXiv:1711.06232, arXiv:1709.04625

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/1912.01452v2)

**Authors**: Jia-Hong Huang, Modar Alfadly, Bernard Ghanem, Marcel Worring

**Abstracts**: Deep neural networks have been playing an essential role in the task of Visual Question Answering (VQA). Until recently, their accuracy has been the main focus of research. Now there is a trend toward assessing the robustness of these models against adversarial attacks by evaluating the accuracy of these models under increasing levels of noisiness in the inputs of VQA models. In VQA, the attack can target the image and/or the proposed query question, dubbed main question, and yet there is a lack of proper analysis of this aspect of VQA. In this work, we propose a new method that uses semantically related questions, dubbed basic questions, acting as noise to evaluate the robustness of VQA models. We hypothesize that as the similarity of a basic question to the main question decreases, the level of noise increases. To generate a reasonable noise level for a given main question, we rank a pool of basic questions based on their similarity with this main question. We cast this ranking problem as a LASSO optimization problem. We also propose a novel robustness measure Rscore and two large-scale basic question datasets in order to standardize robustness analysis of VQA models. The experimental results demonstrate that the proposed evaluation method is able to effectively analyze the robustness of VQA models. To foster the VQA research, we will publish our proposed datasets.



## **2. Detection of Word Adversarial Examples in Text Classification: Benchmark and Baseline via Robust Density Estimation**

cs.CL

Findings of ACL 2022

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.01677v1)

**Authors**: KiYoon Yoo, Jangho Kim, Jiho Jang, Nojun Kwak

**Abstracts**: Word-level adversarial attacks have shown success in NLP models, drastically decreasing the performance of transformer-based models in recent years. As a countermeasure, adversarial defense has been explored, but relatively few efforts have been made to detect adversarial examples. However, detecting adversarial examples may be crucial for automated tasks (e.g. review sentiment analysis) that wish to amass information about a certain population and additionally be a step towards a robust defense system. To this end, we release a dataset for four popular attack methods on four datasets and four models to encourage further research in this field. Along with it, we propose a competitive baseline based on density estimation that has the highest AUC on 29 out of 30 dataset-attack-model combinations. Source code is available in https://github.com/anoymous92874838/text-adv-detection.



## **3. On Improving Adversarial Transferability of Vision Transformers**

cs.CV

ICLR'22 (Spotlight), the first two authors contributed equally. Code:  https://t.ly/hBbW

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2106.04169v3)

**Authors**: Muzammal Naseer, Kanchana Ranasinghe, Salman Khan, Fahad Shahbaz Khan, Fatih Porikli

**Abstracts**: Vision transformers (ViTs) process input images as sequences of patches via self-attention; a radically different architecture than convolutional neural networks (CNNs). This makes it interesting to study the adversarial feature space of ViT models and their transferability. In particular, we observe that adversarial patterns found via conventional adversarial attacks show very \emph{low} black-box transferability even for large ViT models. We show that this phenomenon is only due to the sub-optimal attack procedures that do not leverage the true representation potential of ViTs. A deep ViT is composed of multiple blocks, with a consistent architecture comprising of self-attention and feed-forward layers, where each block is capable of independently producing a class token. Formulating an attack using only the last class token (conventional approach) does not directly leverage the discriminative information stored in the earlier tokens, leading to poor adversarial transferability of ViTs. Using the compositional nature of ViT models, we enhance transferability of existing attacks by introducing two novel strategies specific to the architecture of ViT models. (i) Self-Ensemble: We propose a method to find multiple discriminative pathways by dissecting a single ViT model into an ensemble of networks. This allows explicitly utilizing class-specific information at each ViT block. (ii) Token Refinement: We then propose to refine the tokens to further enhance the discriminative capacity at each block of ViT. Our token refinement systematically combines the class tokens with structural information preserved within the patch tokens.



## **4. On Robustness of Neural Ordinary Differential Equations**

cs.LG

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/1910.05513v4)

**Authors**: Hanshu Yan, Jiawei Du, Vincent Y. F. Tan, Jiashi Feng

**Abstracts**: Neural ordinary differential equations (ODEs) have been attracting increasing attention in various research domains recently. There have been some works studying optimization issues and approximation capabilities of neural ODEs, but their robustness is still yet unclear. In this work, we fill this important gap by exploring robustness properties of neural ODEs both empirically and theoretically. We first present an empirical study on the robustness of the neural ODE-based networks (ODENets) by exposing them to inputs with various types of perturbations and subsequently investigating the changes of the corresponding outputs. In contrast to conventional convolutional neural networks (CNNs), we find that the ODENets are more robust against both random Gaussian perturbations and adversarial attack examples. We then provide an insightful understanding of this phenomenon by exploiting a certain desirable property of the flow of a continuous-time ODE, namely that integral curves are non-intersecting. Our work suggests that, due to their intrinsic robustness, it is promising to use neural ODEs as a basic block for building robust deep network models. To further enhance the robustness of vanilla neural ODEs, we propose the time-invariant steady neural ODE (TisODE), which regularizes the flow on perturbed data via the time-invariant property and the imposition of a steady-state constraint. We show that the TisODE method outperforms vanilla neural ODEs and also can work in conjunction with other state-of-the-art architectural methods to build more robust deep networks.



## **5. Medical Aegis: Robust adversarial protectors for medical images**

cs.CV

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2111.10969v3)

**Authors**: Qingsong Yao, Zecheng He, S. Kevin Zhou

**Abstracts**: Deep neural network based medical image systems are vulnerable to adversarial examples. Many defense mechanisms have been proposed in the literature, however, the existing defenses assume a passive attacker who knows little about the defense system and does not change the attack strategy according to the defense. Recent works have shown that a strong adaptive attack, where an attacker is assumed to have full knowledge about the defense system, can easily bypass the existing defenses. In this paper, we propose a novel adversarial example defense system called Medical Aegis. To the best of our knowledge, Medical Aegis is the first defense in the literature that successfully addresses the strong adaptive adversarial example attacks to medical images. Medical Aegis boasts two-tier protectors: The first tier of Cushion weakens the adversarial manipulation capability of an attack by removing its high-frequency components, yet posing a minimal effect on classification performance of the original image; the second tier of Shield learns a set of per-class DNN models to predict the logits of the protected model. Deviation from the Shield's prediction indicates adversarial examples. Shield is inspired by the observations in our stress tests that there exist robust trails in the shallow layers of a DNN model, which the adaptive attacks can hardly destruct. Experimental results show that the proposed defense accurately detects adaptive attacks, with negligible overhead for model inference.



## **6. Authentication Attacks on Projection-based Cancelable Biometric Schemes**

cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2110.15163v2)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.



## **7. Ad2Attack: Adaptive Adversarial Attack on Real-Time UAV Tracking**

cs.CV

7 pages, 7 figures, accepted by ICRA 2022

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.01516v1)

**Authors**: Changhong Fu, Sihang Li, Xinnan Yuan, Junjie Ye, Ziang Cao, Fangqiang Ding

**Abstracts**: Visual tracking is adopted to extensive unmanned aerial vehicle (UAV)-related applications, which leads to a highly demanding requirement on the robustness of UAV trackers. However, adding imperceptible perturbations can easily fool the tracker and cause tracking failures. This risk is often overlooked and rarely researched at present. Therefore, to help increase awareness of the potential risk and the robustness of UAV tracking, this work proposes a novel adaptive adversarial attack approach, i.e., Ad$^2$Attack, against UAV object tracking. Specifically, adversarial examples are generated online during the resampling of the search patch image, which leads trackers to lose the target in the following frames. Ad$^2$Attack is composed of a direct downsampling module and a super-resolution upsampling module with adaptive stages. A novel optimization function is proposed for balancing the imperceptibility and efficiency of the attack. Comprehensive experiments on several well-known benchmarks and real-world conditions show the effectiveness of our attack method, which dramatically reduces the performance of the most advanced Siamese trackers.



## **8. Label Leakage and Protection from Forward Embedding in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.01451v1)

**Authors**: Jiankai Sun, Xin Yang, Yuanshun Yao, Chong Wang

**Abstracts**: Vertical federated learning (vFL) has gained much attention and been deployed to solve machine learning problems with data privacy concerns in recent years. However, some recent work demonstrated that vFL is vulnerable to privacy leakage even though only the forward intermediate embedding (rather than raw features) and backpropagated gradients (rather than raw labels) are communicated between the involved participants. As the raw labels often contain highly sensitive information, some recent work has been proposed to prevent the label leakage from the backpropagated gradients effectively in vFL. However, these work only identified and defended the threat of label leakage from the backpropagated gradients. None of these work has paid attention to the problem of label leakage from the intermediate embedding. In this paper, we propose a practical label inference method which can steal private labels effectively from the shared intermediate embedding even though some existing protection methods such as label differential privacy and gradients perturbation are applied. The effectiveness of the label attack is inseparable from the correlation between the intermediate embedding and corresponding private labels. To mitigate the issue of label leakage from the forward embedding, we add an additional optimization goal at the label party to limit the label stealing ability of the adversary by minimizing the distance correlation between the intermediate embedding and corresponding private labels. We conducted massive experiments to demonstrate the effectiveness of our proposed protection methods.



## **9. Two Attacks On Proof-of-Stake GHOST/Ethereum**

cs.CR

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.01315v1)

**Authors**: Joachim Neu, Ertem Nusret Tas, David Tse

**Abstracts**: We present two attacks targeting the Proof-of-Stake (PoS) Ethereum consensus protocol. The first attack suggests a fundamental conceptual incompatibility between PoS and the Greedy Heaviest-Observed Sub-Tree (GHOST) fork choice paradigm employed by PoS Ethereum. In a nutshell, PoS allows an adversary with a vanishing amount of stake to produce an unlimited number of equivocating blocks. While most equivocating blocks will be orphaned, such orphaned `uncle blocks' still influence fork choice under the GHOST paradigm, bestowing upon the adversary devastating control over the canonical chain. While the Latest Message Driven (LMD) aspect of current PoS Ethereum prevents a straightforward application of this attack, our second attack shows how LMD specifically can be exploited to obtain a new variant of the balancing attack that overcomes a recent protocol addition that was intended to mitigate balancing-type attacks. Thus, in its current form, PoS Ethereum without and with LMD is vulnerable to our first and second attack, respectively.



## **10. Detecting Adversarial Perturbations in Multi-Task Perception**

cs.CV

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.01177v1)

**Authors**: Marvin Klingner, Varun Ravi Kumar, Senthil Yogamani, Andreas Bär, Tim Fingscheidt

**Abstracts**: While deep neural networks (DNNs) achieve impressive performance on environment perception tasks, their sensitivity to adversarial perturbations limits their use in practical applications. In this paper, we (i) propose a novel adversarial perturbation detection scheme based on multi-task perception of complex vision tasks (i.e., depth estimation and semantic segmentation). Specifically, adversarial perturbations are detected by inconsistencies between extracted edges of the input image, the depth output, and the segmentation output. To further improve this technique, we (ii) develop a novel edge consistency loss between all three modalities, thereby improving their initial consistency which in turn supports our detection scheme. We verify our detection scheme's effectiveness by employing various known attacks and image noises. In addition, we (iii) develop a multi-task adversarial attack, aiming at fooling both tasks as well as our detection scheme. Experimental evaluation on the Cityscapes and KITTI datasets shows that under an assumption of a 5% false positive rate up to 100% of images are correctly detected as adversarially perturbed, depending on the strength of the perturbation. Code will be available on github. A short video at https://youtu.be/KKa6gOyWmH4 provides qualitative results.



## **11. How to Inject Backdoors with Better Consistency: Logit Anchoring on Clean Data**

cs.LG

Accepted by ICLR 2022

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2109.01300v2)

**Authors**: Zhiyuan Zhang, Lingjuan Lyu, Weiqiang Wang, Lichao Sun, Xu Sun

**Abstracts**: Since training a large-scale backdoored model from scratch requires a large training dataset, several recent attacks have considered to inject backdoors into a trained clean model without altering model behaviors on the clean data. Previous work finds that backdoors can be injected into a trained clean model with Adversarial Weight Perturbation (AWP). Here AWPs refers to the variations of parameters that are small in backdoor learning. In this work, we observe an interesting phenomenon that the variations of parameters are always AWPs when tuning the trained clean model to inject backdoors. We further provide theoretical analysis to explain this phenomenon. We formulate the behavior of maintaining accuracy on clean data as the consistency of backdoored models, which includes both global consistency and instance-wise consistency. We extensively analyze the effects of AWPs on the consistency of backdoored models. In order to achieve better consistency, we propose a novel anchoring loss to anchor or freeze the model behaviors on the clean data, with a theoretical guarantee. Both the analytical and the empirical results validate the effectiveness of the anchoring loss in improving the consistency, especially the instance-wise consistency.



## **12. Video is All You Need: Attacking PPG-based Biometric Authentication**

cs.CR

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.00928v1)

**Authors**: Lin Li, Chao Chen, Lei Pan, Jun Zhang, Yang Xiang

**Abstracts**: Unobservable physiological signals enhance biometric authentication systems. Photoplethysmography (PPG) signals are convenient owning to its ease of measurement and are usually well protected against remote adversaries in authentication. Any leaked PPG signals help adversaries compromise the biometric authentication systems, and the advent of remote PPG (rPPG) enables adversaries to acquire PPG signals through restoration. While potentially dangerous, rPPG-based attacks are overlooked because existing methods require the victim's PPG signals. This paper proposes a novel spoofing attack approach that uses the waveforms of rPPG signals extracted from video clips to fool the PPG-based biometric authentication. We develop a new PPG restoration model that does not require leaked PPG signals for adversarial attacks. Test results on state-of-art PPG-based biometric authentication show that the signals recovered through rPPG pose a severe threat to PPG-based biometric authentication.



## **13. Canonical foliations of neural networks: application to robustness**

stat.ML

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.00922v1)

**Authors**: Eliot Tron, Nicolas Couellan, Stéphane Puechmorel

**Abstracts**: Adversarial attack is an emerging threat to the trustability of machine learning. Understanding these attacks is becoming a crucial task. We propose a new vision on neural network robustness using Riemannian geometry and foliation theory, and create a new adversarial attack by taking into account the curvature of the data space. This new adversarial attack called the "dog-leg attack" is a two-step approximation of a geodesic in the data space. The data space is treated as a (pseudo) Riemannian manifold equipped with the pullback of the Fisher Information Metric (FIM) of the neural network. In most cases, this metric is only semi-definite and its kernel becomes a central object to study. A canonical foliation is derived from this kernel. The curvature of the foliation's leaves gives the appropriate correction to get a two-step approximation of the geodesic and hence a new efficient adversarial attack. Our attack is tested on a toy example, a neural network trained to mimic the $\texttt{Xor}$ function, and demonstrates better results that the state of the art attack presented by Zhao et al. (2019).



## **14. MIAShield: Defending Membership Inference Attacks via Preemptive Exclusion of Members**

cs.CR

21 pages, 17 figures, 10 tables

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.00915v1)

**Authors**: Ismat Jarin, Birhanu Eshete

**Abstracts**: In membership inference attacks (MIAs), an adversary observes the predictions of a model to determine whether a sample is part of the model's training data. Existing MIA defenses conceal the presence of a target sample through strong regularization, knowledge distillation, confidence masking, or differential privacy.   We propose MIAShield, a new MIA defense based on preemptive exclusion of member samples instead of masking the presence of a member. The key insight in MIAShield is weakening the strong membership signal that stems from the presence of a target sample by preemptively excluding it at prediction time without compromising model utility. To that end, we design and evaluate a suite of preemptive exclusion oracles leveraging model-confidence, exact or approximate sample signature, and learning-based exclusion of member data points. To be practical, MIAShield splits a training data into disjoint subsets and trains each subset to build an ensemble of models. The disjointedness of subsets ensures that a target sample belongs to only one subset, which isolates the sample to facilitate the preemptive exclusion goal.   We evaluate MIAShield on three benchmark image classification datasets. We show that MIAShield effectively mitigates membership inference (near random guess) for a wide range of MIAs, achieves far better privacy-utility trade-off compared with state-of-the-art defenses, and remains resilient against an adaptive adversary.



## **15. Proceedings of the Artificial Intelligence for Cyber Security (AICS) Workshop at AAAI 2022**

cs.CR

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2202.14010v2)

**Authors**: James Holt, Edward Raff, Ahmad Ridley, Dennis Ross, Arunesh Sinha, Diane Staheli, William Streilen, Milind Tambe, Yevgeniy Vorobeychik, Allan Wollaber

**Abstracts**: The workshop will focus on the application of AI to problems in cyber security. Cyber systems generate large volumes of data, utilizing this effectively is beyond human capabilities. Additionally, adversaries continue to develop new attacks. Hence, AI methods are required to understand and protect the cyber domain. These challenges are widely studied in enterprise networks, but there are many gaps in research and practice as well as novel problems in other domains.   In general, AI techniques are still not widely adopted in the real world. Reasons include: (1) a lack of certification of AI for security, (2) a lack of formal study of the implications of practical constraints (e.g., power, memory, storage) for AI systems in the cyber domain, (3) known vulnerabilities such as evasion, poisoning attacks, (4) lack of meaningful explanations for security analysts, and (5) lack of analyst trust in AI solutions. There is a need for the research community to develop novel solutions for these practical issues.



## **16. Beyond Gradients: Exploiting Adversarial Priors in Model Inversion Attacks**

cs.LG

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2203.00481v1)

**Authors**: Dmitrii Usynin, Daniel Rueckert, Georgios Kaissis

**Abstracts**: Collaborative machine learning settings like federated learning can be susceptible to adversarial interference and attacks. One class of such attacks is termed model inversion attacks, characterised by the adversary reverse-engineering the model to extract representations and thus disclose the training data. Prior implementations of this attack typically only rely on the captured data (i.e. the shared gradients) and do not exploit the data the adversary themselves control as part of the training consortium. In this work, we propose a novel model inversion framework that builds on the foundations of gradient-based model inversion attacks, but additionally relies on matching the features and the style of the reconstructed image to data that is controlled by an adversary. Our technique outperforms existing gradient-based approaches both qualitatively and quantitatively, while still maintaining the same honest-but-curious threat model, allowing the adversary to obtain enhanced reconstructions while remaining concealed.



## **17. RAB: Provable Robustness Against Backdoor Attacks**

cs.LG

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2003.08904v6)

**Authors**: Maurice Weber, Xiaojun Xu, Bojan Karlaš, Ce Zhang, Bo Li

**Abstracts**: Recent studies have shown that deep neural networks are vulnerable to adversarial attacks, including evasion and backdoor (poisoning) attacks. On the defense side, there have been intensive efforts on improving both empirical and provable robustness against evasion attacks; however, provable robustness against backdoor attacks still remains largely unexplored. In this paper, we focus on certifying the machine learning model robustness against general threat models, especially backdoor attacks. We first provide a unified framework via randomized smoothing techniques and show how it can be instantiated to certify the robustness against both evasion and backdoor attacks. We then propose the first robust training process, RAB, to smooth the trained model and certify its robustness against backdoor attacks. We theoretically prove the robustness bound for machine learning models trained with RAB, and prove that our robustness bound is tight. We derive the robustness conditions for different smoothing distributions including Gaussian and uniform distributions. In addition, we theoretically show that it is possible to train the robust smoothed models efficiently for simple models such as K-nearest neighbor classifiers, and we propose an exact smooth-training algorithm which eliminates the need to sample from a noise distribution for such models. Empirically, we conduct comprehensive experiments for different machine learning models such as DNNs and K-NN models on MNIST, CIFAR-10, and ImageNette datasets and provide the first benchmark for certified robustness against backdoor attacks. In addition, we evaluate K-NN models on a spambase tabular dataset to demonstrate the advantages of the proposed exact algorithm. Both the theoretic analysis and the comprehensive evaluation on diverse ML models and datasets shed lights on further robust learning strategies against general training time attacks.



## **18. Adversarial samples for deep monocular 6D object pose estimation**

cs.CV

15 pages. arXiv admin note: text overlap with arXiv:2105.14291 by  other authors

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2203.00302v1)

**Authors**: Jinlai Zhang, Weiming Li, Shuang Liang, Hao Wang, Jihong Zhu

**Abstracts**: Estimating object 6D pose from an RGB image is important for many real-world applications such as autonomous driving and robotic grasping, where robustness of the estimation is crucial. In this work, for the first time, we study adversarial samples that can fool state-of-the-art (SOTA) deep learning based 6D pose estimation models. In particular, we propose a Unified 6D pose estimation Attack, namely U6DA, which can successfully attack all the three main categories of models for 6D pose estimation. The key idea of our U6DA is to fool the models to predict wrong results for object shapes that are essential for correct 6D pose estimation. Specifically, we explore a transfer-based black-box attack to 6D pose estimation. By shifting the segmentation attention map away from its original position, adversarial samples are crafted. We show that such adversarial samples are not only effective for the direct 6D pose estimation models, but also able to attack the two-stage based models regardless of their robust RANSAC modules. Extensive experiments were conducted to demonstrate the effectiveness of our U6DA with large-scale public benchmarks. We also introduce a new U6DA-Linemod dataset for robustness study of the 6D pose estimation task. Our codes and dataset will be available at \url{https://github.com/cuge1995/U6DA}.



## **19. Towards Robust Stacked Capsule Autoencoder with Hybrid Adversarial Training**

cs.CV

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2202.13755v2)

**Authors**: Jiazhu Dai, Siwei Xiong

**Abstracts**: Capsule networks (CapsNets) are new neural networks that classify images based on the spatial relationships of features. By analyzing the pose of features and their relative positions, it is more capable to recognize images after affine transformation. The stacked capsule autoencoder (SCAE) is a state-of-the-art CapsNet, and achieved unsupervised classification of CapsNets for the first time. However, the security vulnerabilities and the robustness of the SCAE has rarely been explored. In this paper, we propose an evasion attack against SCAE, where the attacker can generate adversarial perturbations based on reducing the contribution of the object capsules in SCAE related to the original category of the image. The adversarial perturbations are then applied to the original images, and the perturbed images will be misclassified. Furthermore, we propose a defense method called Hybrid Adversarial Training (HAT) against such evasion attacks. HAT makes use of adversarial training and adversarial distillation to achieve better robustness and stability. We evaluate the defense method and the experimental results show that the refined SCAE model can achieve 82.14% classification accuracy under evasion attack. The source code is available at https://github.com/FrostbiteXSW/SCAE_Defense.



## **20. Towards Effective and Robust Neural Trojan Defenses via Input Filtering**

cs.CR

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2202.12154v2)

**Authors**: Kien Do, Haripriya Harikumar, Hung Le, Dung Nguyen, Truyen Tran, Santu Rana, Dang Nguyen, Willy Susilo, Svetha Venkatesh

**Abstracts**: Trojan attacks on deep neural networks are both dangerous and surreptitious. Over the past few years, Trojan attacks have advanced from using only a simple trigger and targeting only one class to using many sophisticated triggers and targeting multiple classes. However, Trojan defenses have not caught up with this development. Most defense methods still make out-of-date assumptions about Trojan triggers and target classes, thus, can be easily circumvented by modern Trojan attacks. In this paper, we advocate general defenses that are effective and robust against various Trojan attacks and propose two novel "filtering" defenses with these characteristics called Variational Input Filtering (VIF) and Adversarial Input Filtering (AIF). VIF and AIF leverage variational inference and adversarial training respectively to purify all potential Trojan triggers in the input at run time without making any assumption about their numbers and forms. We further extend "filtering" to "filtering-then-contrasting" - a new defense mechanism that helps avoid the drop in classification accuracy on clean data caused by filtering. Extensive experimental results show that our proposed defenses significantly outperform 4 well-known defenses in mitigating 5 different Trojan attacks including the two state-of-the-art which defeat many strong defenses.



## **21. Adversarial Attack Framework on Graph Embedding Models with Limited Knowledge**

cs.LG

Journal extension of GF-Attack, accepted by TKDE

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2105.12419v2)

**Authors**: Heng Chang, Yu Rong, Tingyang Xu, Wenbing Huang, Honglei Zhang, Peng Cui, Xin Wang, Wenwu Zhu, Junzhou Huang

**Abstracts**: With the success of the graph embedding model in both academic and industry areas, the robustness of graph embedding against adversarial attack inevitably becomes a crucial problem in graph learning. Existing works usually perform the attack in a white-box fashion: they need to access the predictions/labels to construct their adversarial loss. However, the inaccessibility of predictions/labels makes the white-box attack impractical to a real graph learning system. This paper promotes current frameworks in a more general and flexible sense -- we demand to attack various kinds of graph embedding models with black-box driven. We investigate the theoretical connections between graph signal processing and graph embedding models and formulate the graph embedding model as a general graph signal process with a corresponding graph filter. Therefore, we design a generalized adversarial attacker: GF-Attack. Without accessing any labels and model predictions, GF-Attack can perform the attack directly on the graph filter in a black-box fashion. We further prove that GF-Attack can perform an effective attack without knowing the number of layers of graph embedding models. To validate the generalization of GF-Attack, we construct the attacker on four popular graph embedding models. Extensive experiments validate the effectiveness of GF-Attack on several benchmark datasets.



## **22. Load-Altering Attacks Against Power Grids under COVID-19 Low-Inertia Conditions**

cs.CR

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2201.10505v2)

**Authors**: Subhash Lakshminarayana, Juan Ospina, Charalambos Konstantinou

**Abstracts**: The COVID-19 pandemic has impacted our society by forcing shutdowns and shifting the way people interacted worldwide. In relation to the impacts on the electric grid, it created a significant decrease in energy demands across the globe. Recent studies have shown that the low demand conditions caused by COVID-19 lockdowns combined with large renewable generation have resulted in extremely low-inertia grid conditions. In this work, we examine how an attacker could exploit these {scenarios} to cause unsafe grid operating conditions by executing load-altering attacks (LAAs) targeted at compromising hundreds of thousands of IoT-connected high-wattage loads in low-inertia power systems. Our study focuses on analyzing the impact of the COVID-19 mitigation measures on U.S. regional transmission operators (RTOs), formulating a plausible and realistic least-effort LAA targeted at transmission systems with low-inertia conditions, and evaluating the probability of these large-scale LAAs. Theoretical and simulation results are presented based on the WSCC 9-bus {and IEEE 118-bus} test systems. Results demonstrate how adversaries could provoke major frequency disturbances by targeting vulnerable load buses in low-inertia systems and offer insights into how the temporal fluctuations of renewable energy sources, considering generation scheduling, impact the grid's vulnerability to LAAs.



## **23. MaMaDroid2.0 -- The Holes of Control Flow Graphs**

cs.CR

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13922v1)

**Authors**: Harel Berger, Chen Hajaj, Enrico Mariconti, Amit Dvir

**Abstracts**: Android malware is a continuously expanding threat to billions of mobile users around the globe. Detection systems are updated constantly to address these threats. However, a backlash takes the form of evasion attacks, in which an adversary changes malicious samples such that those samples will be misclassified as benign. This paper fully inspects a well-known Android malware detection system, MaMaDroid, which analyzes the control flow graph of the application. Changes to the portion of benign samples in the train set and models are considered to see their effect on the classifier. The changes in the ratio between benign and malicious samples have a clear effect on each one of the models, resulting in a decrease of more than 40% in their detection rate. Moreover, adopted ML models are implemented as well, including 5-NN, Decision Tree, and Adaboost. Exploration of the six models reveals a typical behavior in different cases, of tree-based models and distance-based models. Moreover, three novel attacks that manipulate the CFG and their detection rates are described for each one of the targeted models. The attacks decrease the detection rate of most of the models to 0%, with regards to different ratios of benign to malicious apps. As a result, a new version of MaMaDroid is engineered. This model fuses the CFG of the app and static analysis of features of the app. This improved model is proved to be robust against evasion attacks targeting both CFG-based models and static analysis models, achieving a detection rate of more than 90% against each one of the attacks.



## **24. Formally verified asymptotic consensus in robust networks**

cs.PL

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13833v1)

**Authors**: Mohit Tekriwal, Avi Tachna-Fram, Jean-Baptiste Jeannin, Manos Kapritsos, Dimitra Panagou

**Abstracts**: Distributed architectures are used to improve performance and reliability of various systems. An important capability of a distributed architecture is the ability to reach consensus among all its nodes. To achieve this, several consensus algorithms have been proposed for various scenarii, and many of these algorithms come with proofs of correctness that are not mechanically checked. Unfortunately, those proofs are known to be intricate and prone to errors.   In this paper, we formalize and mechanically check a consensus algorithm widely used in the distributed controls community: the Weighted-Mean Subsequence Reduced (W-MSR) algorithm proposed by Le Blanc et al. This algorithm provides a way to achieve asymptotic consensus in a distributed controls scenario in the presence of adversarial agents (attackers) that may not update their states based on the nominal consensus protocol, and may share inaccurate information with their neighbors. Using the Coq proof assistant, we formalize the necessary and sufficient conditions required to achieve resilient asymptotic consensus under the assumed attacker model. We leverage the existing Coq formalizations of graph theory, finite sets and sequences of the mathcomp library for our development. To our knowledge, this is the first mechanical proof of an asymptotic consensus algorithm. During the formalization, we clarify several imprecisions in the paper proof, including an imprecision on quantifiers in the main theorem.



## **25. Robust Textual Embedding against Word-level Adversarial Attacks**

cs.CL

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13817v1)

**Authors**: Yichen Yang, Xiaosen Wang, Kun He

**Abstracts**: We attribute the vulnerability of natural language processing models to the fact that similar inputs are converted to dissimilar representations in the embedding space, leading to inconsistent outputs, and propose a novel robust training method, termed Fast Triplet Metric Learning (FTML). Specifically, we argue that the original sample should have similar representation with its adversarial counterparts and distinguish its representation from other samples for better robustness. To this end, we adopt the triplet metric learning into the standard training to pull the words closer to their positive samples (i.e., synonyms) and push away their negative samples (i.e., non-synonyms) in the embedding space. Extensive experiments demonstrate that FTML can significantly promote the model robustness against various advanced adversarial attacks while keeping competitive classification accuracy on original samples. Besides, our method is efficient as it only needs to adjust the embedding and introduces very little overhead on the standard training. Our work shows the great potential of improving the textual robustness through robust word embedding.



## **26. On the Robustness of CountSketch to Adaptive Inputs**

cs.DS

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13736v1)

**Authors**: Edith Cohen, Xin Lyu, Jelani Nelson, Tamás Sarlós, Moshe Shechner, Uri Stemmer

**Abstracts**: CountSketch is a popular dimensionality reduction technique that maps vectors to a lower dimension using randomized linear measurements. The sketch supports recovering $\ell_2$-heavy hitters of a vector (entries with $v[i]^2 \geq \frac{1}{k}\|\boldsymbol{v}\|^2_2$). We study the robustness of the sketch in adaptive settings where input vectors may depend on the output from prior inputs. Adaptive settings arise in processes with feedback or with adversarial attacks. We show that the classic estimator is not robust, and can be attacked with a number of queries of the order of the sketch size. We propose a robust estimator (for a slightly modified sketch) that allows for quadratic number of queries in the sketch size, which is an improvement factor of $\sqrt{k}$ (for $k$ heavy hitters) over prior work.



## **27. An Empirical Study on the Intrinsic Privacy of SGD**

cs.LG

21 pages, 11 figures, 8 tables

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/1912.02919v4)

**Authors**: Stephanie L. Hyland, Shruti Tople

**Abstracts**: Introducing noise in the training of machine learning systems is a powerful way to protect individual privacy via differential privacy guarantees, but comes at a cost to utility. This work looks at whether the inherent randomness of stochastic gradient descent (SGD) could contribute to privacy, effectively reducing the amount of \emph{additional} noise required to achieve a given privacy guarantee. We conduct a large-scale empirical study to examine this question. Training a grid of over 120,000 models across four datasets (tabular and images) on convex and non-convex objectives, we demonstrate that the random seed has a larger impact on model weights than any individual training example. We test the distribution over weights induced by the seed, finding that the simple convex case can be modelled with a multivariate Gaussian posterior, while neural networks exhibit multi-modal and non-Gaussian weight distributions. By casting convex SGD as a Gaussian mechanism, we then estimate an `intrinsic' data-dependent $\epsilon_i(\mathcal{D})$, finding values as low as 6.3, dropping to 1.9 using empirical estimates. We use a membership inference attack to estimate $\epsilon$ for non-convex SGD and demonstrate that hiding the random seed from the adversary results in a statistically significant reduction in attack performance, corresponding to a reduction in the effective $\epsilon$. These results provide empirical evidence that SGD exhibits appreciable variability relative to its dataset sensitivity, and this `intrinsic noise' has the potential to be leveraged to improve the utility of privacy-preserving machine learning.



## **28. Enhance transferability of adversarial examples with model architecture**

cs.LG

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13625v1)

**Authors**: Mingyuan Fan, Wenzhong Guo, Shengxing Yu, Zuobin Ying, Ximeng Liu

**Abstracts**: Transferability of adversarial examples is of critical importance to launch black-box adversarial attacks, where attackers are only allowed to access the output of the target model. However, under such a challenging but practical setting, the crafted adversarial examples are always prone to overfitting to the proxy model employed, presenting poor transferability. In this paper, we suggest alleviating the overfitting issue from a novel perspective, i.e., designing a fitted model architecture. Specifically, delving the bottom of the cause of poor transferability, we arguably decompose and reconstruct the existing model architecture into an effective model architecture, namely multi-track model architecture (MMA). The adversarial examples crafted on the MMA can maximumly relieve the effect of model-specified features to it and toward the vulnerable directions adopted by diverse architectures. Extensive experimental evaluation demonstrates that the transferability of adversarial examples based on the MMA significantly surpass other state-of-the-art model architectures by up to 40% with comparable overhead.



## **29. GRAPHITE: Generating Automatic Physical Examples for Machine-Learning Attacks on Computer Vision Systems**

cs.CR

IEEE European Symposium on Security and Privacy 2022 (EuroS&P 2022)

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2002.07088v6)

**Authors**: Ryan Feng, Neal Mangaokar, Jiefeng Chen, Earlence Fernandes, Somesh Jha, Atul Prakash

**Abstracts**: This paper investigates an adversary's ease of attack in generating adversarial examples for real-world scenarios. We address three key requirements for practical attacks for the real-world: 1) automatically constraining the size and shape of the attack so it can be applied with stickers, 2) transform-robustness, i.e., robustness of a attack to environmental physical variations such as viewpoint and lighting changes, and 3) supporting attacks in not only white-box, but also black-box hard-label scenarios, so that the adversary can attack proprietary models. In this work, we propose GRAPHITE, an efficient and general framework for generating attacks that satisfy the above three key requirements. GRAPHITE takes advantage of transform-robustness, a metric based on expectation over transforms (EoT), to automatically generate small masks and optimize with gradient-free optimization. GRAPHITE is also flexible as it can easily trade-off transform-robustness, perturbation size, and query count in black-box settings. On a GTSRB model in a hard-label black-box setting, we are able to find attacks on all possible 1,806 victim-target class pairs with averages of 77.8% transform-robustness, perturbation size of 16.63% of the victim images, and 126K queries per pair. For digital-only attacks where achieving transform-robustness is not a requirement, GRAPHITE is able to find successful small-patch attacks with an average of only 566 queries for 92.2% of victim-target pairs. GRAPHITE is also able to find successful attacks using perturbations that modify small areas of the input image against PatchGuard, a recently proposed defense against patch-based attacks.



## **30. Markov Chain Monte Carlo-Based Machine Unlearning: Unlearning What Needs to be Forgotten**

cs.LG

Proceedings of the 2022 ACM Asia Conference on Computer and  Communications Security (ASIA CCS '22), May 30-June 3, 2022, Nagasaki, Japan

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13585v1)

**Authors**: Quoc Phong Nguyen, Ryutaro Oikawa, Dinil Mon Divakaran, Mun Choon Chan, Bryan Kian Hsiang Low

**Abstracts**: As the use of machine learning (ML) models is becoming increasingly popular in many real-world applications, there are practical challenges that need to be addressed for model maintenance. One such challenge is to 'undo' the effect of a specific subset of dataset used for training a model. This specific subset may contain malicious or adversarial data injected by an attacker, which affects the model performance. Another reason may be the need for a service provider to remove data pertaining to a specific user to respect the user's privacy. In both cases, the problem is to 'unlearn' a specific subset of the training data from a trained model without incurring the costly procedure of retraining the whole model from scratch. Towards this goal, this paper presents a Markov chain Monte Carlo-based machine unlearning (MCU) algorithm. MCU helps to effectively and efficiently unlearn a trained model from subsets of training dataset. Furthermore, we show that with MCU, we are able to explain the effect of a subset of a training dataset on the model prediction. Thus, MCU is useful for examining subsets of data to identify the adversarial data to be removed. Similarly, MCU can be used to erase the lineage of a user's personal data from trained ML models, thus upholding a user's "right to be forgotten". We empirically evaluate the performance of our proposed MCU algorithm on real-world phishing and diabetes datasets. Results show that MCU can achieve a desirable performance by efficiently removing the effect of a subset of training dataset and outperform an existing algorithm that utilizes the remaining dataset.



## **31. A Unified Wasserstein Distributional Robustness Framework for Adversarial Training**

cs.LG

**SubmitDate**: 2022-02-27    [paper-pdf](http://arxiv.org/pdf/2202.13437v1)

**Authors**: Tuan Anh Bui, Trung Le, Quan Tran, He Zhao, Dinh Phung

**Abstracts**: It is well-known that deep neural networks (DNNs) are susceptible to adversarial attacks, exposing a severe fragility of deep learning systems. As the result, adversarial training (AT) method, by incorporating adversarial examples during training, represents a natural and effective approach to strengthen the robustness of a DNN-based classifier. However, most AT-based methods, notably PGD-AT and TRADES, typically seek a pointwise adversary that generates the worst-case adversarial example by independently perturbing each data sample, as a way to "probe" the vulnerability of the classifier. Arguably, there are unexplored benefits in considering such adversarial effects from an entire distribution. To this end, this paper presents a unified framework that connects Wasserstein distributional robustness with current state-of-the-art AT methods. We introduce a new Wasserstein cost function and a new series of risk functions, with which we show that standard AT methods are special cases of their counterparts in our framework. This connection leads to an intuitive relaxation and generalization of existing AT methods and facilitates the development of a new family of distributional robustness AT-based algorithms. Extensive experiments show that our distributional robustness AT algorithms robustify further their standard AT counterparts in various settings.



## **32. Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks**

cs.CV

Accepted at NeurIPS 2021. The missing square term in Eqn.(13), as  well as many other mistakes of the previous version, have been fixed in the  current version

**SubmitDate**: 2022-02-27    [paper-pdf](http://arxiv.org/pdf/2111.07492v5)

**Authors**: Chen Ma, Xiangyu Guo, Li Chen, Jun-Hai Yong, Yisen Wang

**Abstracts**: One major problem in black-box adversarial attacks is the high query complexity in the hard-label attack setting, where only the top-1 predicted label is available. In this paper, we propose a novel geometric-based approach called Tangent Attack (TA), which identifies an optimal tangent point of a virtual hemisphere located on the decision boundary to reduce the distortion of the attack. Assuming the decision boundary is locally flat, we theoretically prove that the minimum $\ell_2$ distortion can be obtained by reaching the decision boundary along the tangent line passing through such tangent point in each iteration. To improve the robustness of our method, we further propose a generalized method which replaces the hemisphere with a semi-ellipsoid to adapt to curved decision boundaries. Our approach is free of pre-training. Extensive experiments conducted on the ImageNet and CIFAR-10 datasets demonstrate that our approach can consume only a small number of queries to achieve the low-magnitude distortion. The implementation source code is released online at https://github.com/machanic/TangentAttack.



## **33. CC-Cert: A Probabilistic Approach to Certify General Robustness of Neural Networks**

cs.LG

In Proceedings of AAAI-22, the Thirty-Sixth AAAI Conference on  Artificial Intelligence

**SubmitDate**: 2022-02-27    [paper-pdf](http://arxiv.org/pdf/2109.10696v2)

**Authors**: Mikhail Pautov, Nurislam Tursynbek, Marina Munkhoeva, Nikita Muravev, Aleksandr Petiushko, Ivan Oseledets

**Abstracts**: In safety-critical machine learning applications, it is crucial to defend models against adversarial attacks -- small modifications of the input that change the predictions. Besides rigorously studied $\ell_p$-bounded additive perturbations, recently proposed semantic perturbations (e.g. rotation, translation) raise a serious concern on deploying ML systems in real-world. Therefore, it is important to provide provable guarantees for deep learning models against semantically meaningful input transformations. In this paper, we propose a new universal probabilistic certification approach based on Chernoff-Cramer bounds that can be used in general attack settings. We estimate the probability of a model to fail if the attack is sampled from a certain distribution. Our theoretical findings are supported by experimental results on different datasets.



## **34. Socialbots on Fire: Modeling Adversarial Behaviors of Socialbots via Multi-Agent Hierarchical Reinforcement Learning**

cs.SI

Accepted to The ACM Web Conference 2022

**SubmitDate**: 2022-02-26    [paper-pdf](http://arxiv.org/pdf/2110.10655v2)

**Authors**: Thai Le, Long Tran-Thanh, Dongwon Lee

**Abstracts**: Socialbots are software-driven user accounts on social platforms, acting autonomously (mimicking human behavior), with the aims to influence the opinions of other users or spread targeted misinformation for particular goals. As socialbots undermine the ecosystem of social platforms, they are often considered harmful. As such, there have been several computational efforts to auto-detect the socialbots. However, to our best knowledge, the adversarial nature of these socialbots has not yet been studied. This begs a question "can adversaries, controlling socialbots, exploit AI techniques to their advantage?" To this question, we successfully demonstrate that indeed it is possible for adversaries to exploit computational learning mechanism such as reinforcement learning (RL) to maximize the influence of socialbots while avoiding being detected. We first formulate the adversarial socialbot learning as a cooperative game between two functional hierarchical RL agents. While one agent curates a sequence of activities that can avoid the detection, the other agent aims to maximize network influence by selectively connecting with right users. Our proposed policy networks train with a vast amount of synthetic graphs and generalize better than baselines on unseen real-life graphs both in terms of maximizing network influence (up to +18%) and sustainable stealthiness (up to +40% undetectability) under a strong bot detector (with 90% detection accuracy). During inference, the complexity of our approach scales linearly, independent of a network's structure and the virality of news. This makes our approach a practical adversarial attack when deployed in a real-life setting.



## **35. Natural Attack for Pre-trained Models of Code**

cs.SE

To appear in the Technical Track of ICSE 2022

**SubmitDate**: 2022-02-26    [paper-pdf](http://arxiv.org/pdf/2201.08698v2)

**Authors**: Zhou Yang, Jieke Shi, Junda He, David Lo

**Abstracts**: Pre-trained models of code have achieved success in many important software engineering tasks. However, these powerful models are vulnerable to adversarial attacks that slightly perturb model inputs to make a victim model produce wrong outputs. Current works mainly attack models of code with examples that preserve operational program semantics but ignore a fundamental requirement for adversarial example generation: perturbations should be natural to human judges, which we refer to as naturalness requirement.   In this paper, we propose ALERT (nAturaLnEss AwaRe ATtack), a black-box attack that adversarially transforms inputs to make victim models produce wrong outputs. Different from prior works, this paper considers the natural semantic of generated examples at the same time as preserving the operational semantic of original inputs. Our user study demonstrates that human developers consistently consider that adversarial examples generated by ALERT are more natural than those generated by the state-of-the-art work by Zhang et al. that ignores the naturalness requirement. On attacking CodeBERT, our approach can achieve attack success rates of 53.62%, 27.79%, and 35.78% across three downstream tasks: vulnerability prediction, clone detection and code authorship attribution. On GraphCodeBERT, our approach can achieve average success rates of 76.95%, 7.96% and 61.47% on the three tasks. The above outperforms the baseline by 14.07% and 18.56% on the two pre-trained models on average. Finally, we investigated the value of the generated adversarial examples to harden victim models through an adversarial fine-tuning procedure and demonstrated the accuracy of CodeBERT and GraphCodeBERT against ALERT-generated adversarial examples increased by 87.59% and 92.32%, respectively.



## **36. Projective Ranking-based GNN Evasion Attacks**

cs.LG

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12993v1)

**Authors**: He Zhang, Xingliang Yuan, Chuan Zhou, Shirui Pan

**Abstracts**: Graph neural networks (GNNs) offer promising learning methods for graph-related tasks. However, GNNs are at risk of adversarial attacks. Two primary limitations of the current evasion attack methods are highlighted: (1) The current GradArgmax ignores the "long-term" benefit of the perturbation. It is faced with zero-gradient and invalid benefit estimates in certain situations. (2) In the reinforcement learning-based attack methods, the learned attack strategies might not be transferable when the attack budget changes. To this end, we first formulate the perturbation space and propose an evaluation framework and the projective ranking method. We aim to learn a powerful attack strategy then adapt it as little as possible to generate adversarial samples under dynamic budget settings. In our method, based on mutual information, we rank and assess the attack benefits of each perturbation for an effective attack strategy. By projecting the strategy, our method dramatically minimizes the cost of learning a new attack strategy when the attack budget changes. In the comparative assessment with GradArgmax and RL-S2V, the results show our method owns high attack performance and effective transferability. The visualization of our method also reveals various attack patterns in the generation of adversarial samples.



## **37. Attacks and Faults Injection in Self-Driving Agents on the Carla Simulator -- Experience Report**

cs.AI

submitted version; appeared at: International Conference on Computer  Safety, Reliability, and Security. Springer, Cham, 2021

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12991v1)

**Authors**: Niccolò Piazzesi, Massimo Hong, Andrea Ceccarelli

**Abstracts**: Machine Learning applications are acknowledged at the foundation of autonomous driving, because they are the enabling technology for most driving tasks. However, the inclusion of trained agents in automotive systems exposes the vehicle to novel attacks and faults, that can result in safety threats to the driv-ing tasks. In this paper we report our experimental campaign on the injection of adversarial attacks and software faults in a self-driving agent running in a driving simulator. We show that adversarial attacks and faults injected in the trained agent can lead to erroneous decisions and severely jeopardize safety. The paper shows a feasible and easily-reproducible approach based on open source simula-tor and tools, and the results clearly motivate the need of both protective measures and extensive testing campaigns.



## **38. Does Label Differential Privacy Prevent Label Inference Attacks?**

cs.LG

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12968v1)

**Authors**: Ruihan Wu, Jin Peng Zhou, Kilian Q. Weinberger, Chuan Guo

**Abstracts**: Label differential privacy (LDP) is a popular framework for training private ML models on datasets with public features and sensitive private labels. Despite its rigorous privacy guarantee, it has been observed that in practice LDP does not preclude label inference attacks (LIAs): Models trained with LDP can be evaluated on the public training features to recover, with high accuracy, the very private labels that it was designed to protect. In this work, we argue that this phenomenon is not paradoxical and that LDP merely limits the advantage of an LIA adversary compared to predicting training labels using the Bayes classifier. At LDP $\epsilon=0$ this advantage is zero, hence the optimal attack is to predict according to the Bayes classifier and is independent of the training labels. Finally, we empirically demonstrate that our result closely captures the behavior of simulated attacks on both synthetic and real world datasets.



## **39. Robust and Accurate Authorship Attribution via Program Normalization**

cs.LG

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2007.00772v3)

**Authors**: Yizhen Wang, Mohannad Alhanahnah, Ke Wang, Mihai Christodorescu, Somesh Jha

**Abstracts**: Source code attribution approaches have achieved remarkable accuracy thanks to the rapid advances in deep learning. However, recent studies shed light on their vulnerability to adversarial attacks. In particular, they can be easily deceived by adversaries who attempt to either create a forgery of another author or to mask the original author. To address these emerging issues, we formulate this security challenge into a general threat model, the $\textit{relational adversary}$, that allows an arbitrary number of the semantics-preserving transformations to be applied to an input in any problem space. Our theoretical investigation shows the conditions for robustness and the trade-off between robustness and accuracy in depth. Motivated by these insights, we present a novel learning framework, $\textit{normalize-and-predict}$ ($\textit{N&P}$), that in theory guarantees the robustness of any authorship-attribution approach. We conduct an extensive evaluation of $\textit{N&P}$ in defending two of the latest authorship-attribution approaches against state-of-the-art attack methods. Our evaluation demonstrates that $\textit{N&P}$ improves the accuracy on adversarial inputs by as much as 70% over the vanilla models. More importantly, $\textit{N&P}$ also increases robust accuracy to 45% higher than adversarial training while running over 40 times faster.



## **40. ARIA: Adversarially Robust Image Attribution for Content Provenance**

cs.CV

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12860v1)

**Authors**: Maksym Andriushchenko, Xiaoyang Rebecca Li, Geoffrey Oxholm, Thomas Gittings, Tu Bui, Nicolas Flammarion, John Collomosse

**Abstracts**: Image attribution -- matching an image back to a trusted source -- is an emerging tool in the fight against online misinformation. Deep visual fingerprinting models have recently been explored for this purpose. However, they are not robust to tiny input perturbations known as adversarial examples. First we illustrate how to generate valid adversarial images that can easily cause incorrect image attribution. Then we describe an approach to prevent imperceptible adversarial attacks on deep visual fingerprinting models, via robust contrastive learning. The proposed training procedure leverages training on $\ell_\infty$-bounded adversarial examples, it is conceptually simple and incurs only a small computational overhead. The resulting models are substantially more robust, are accurate even on unperturbed images, and perform well even over a database with millions of images. In particular, we achieve 91.6% standard and 85.1% adversarial recall under $\ell_\infty$-bounded perturbations on manipulated images compared to 80.1% and 0.0% from prior work. We also show that robustness generalizes to other types of imperceptible perturbations unseen during training. Finally, we show how to train an adversarially robust image comparator model for detecting editorial changes in matched images.



## **41. Short Paper: Device- and Locality-Specific Fingerprinting of Shared NISQ Quantum Computers**

cs.CR

5 pages, 8 figures, HASP 2021 author version

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12731v1)

**Authors**: Allen Mi, Shuwen Deng, Jakub Szefer

**Abstracts**: Fingerprinting of quantum computer devices is a new threat that poses a challenge to shared, cloud-based quantum computers. Fingerprinting can allow adversaries to map quantum computer infrastructures, uniquely identify cloud-based devices which otherwise have no public identifiers, and it can assist other adversarial attacks. This work shows idle tomography-based fingerprinting method based on crosstalk-induced errors in NISQ quantum computers. The device- and locality-specific fingerprinting results show prediction accuracy values of $99.1\%$ and $95.3\%$, respectively.



## **42. Detection as Regression: Certified Object Detection by Median Smoothing**

cs.CV

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2007.03730v4)

**Authors**: Ping-yeh Chiang, Michael J. Curry, Ahmed Abdelkader, Aounon Kumar, John Dickerson, Tom Goldstein

**Abstracts**: Despite the vulnerability of object detectors to adversarial attacks, very few defenses are known to date. While adversarial training can improve the empirical robustness of image classifiers, a direct extension to object detection is very expensive. This work is motivated by recent progress on certified classification by randomized smoothing. We start by presenting a reduction from object detection to a regression problem. Then, to enable certified regression, where standard mean smoothing fails, we propose median smoothing, which is of independent interest. We obtain the first model-agnostic, training-free, and certified defense for object detection against $\ell_2$-bounded attacks. The code for all experiments in the paper is available at http://github.com/Ping-C/CertifiedObjectDetection .



## **43. On the Effectiveness of Dataset Watermarking in Adversarial Settings**

cs.CR

7 pages, 2 figures. Will appear in the proceedings of CODASPY-IWSPA  2022

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12506v1)

**Authors**: Buse Gul Atli Tekgul, N. Asokan

**Abstracts**: In a data-driven world, datasets constitute a significant economic value. Dataset owners who spend time and money to collect and curate the data are incentivized to ensure that their datasets are not used in ways that they did not authorize. When such misuse occurs, dataset owners need technical mechanisms for demonstrating their ownership of the dataset in question. Dataset watermarking provides one approach for ownership demonstration which can, in turn, deter unauthorized use. In this paper, we investigate a recently proposed data provenance method, radioactive data, to assess if it can be used to demonstrate ownership of (image) datasets used to train machine learning (ML) models. The original paper reported that radioactive data is effective in white-box settings. We show that while this is true for large datasets with many classes, it is not as effective for datasets where the number of classes is low $(\leq 30)$ or the number of samples per class is low $(\leq 500)$. We also show that, counter-intuitively, the black-box verification technique is effective for all datasets used in this paper, even when white-box verification is not. Given this observation, we show that the confidence in white-box verification can be improved by using watermarked samples directly during the verification process. We also highlight the need to assess the robustness of radioactive data if it were to be used for ownership demonstration since it is an adversarial setting unlike provenance identification.   Compared to dataset watermarking, ML model watermarking has been explored more extensively in recent literature. However, most of the model watermarking techniques can be defeated via model extraction. We show that radioactive data can effectively survive model extraction attacks, which raises the possibility that it can be used for ML model ownership verification robust against model extraction.



## **44. Understanding Adversarial Robustness from Feature Maps of Convolutional Layers**

cs.CV

10pages

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12435v1)

**Authors**: Cong Xu, Min Yang

**Abstracts**: The adversarial robustness of a neural network mainly relies on two factors, one is the feature representation capacity of the network, and the other is its resistance ability to perturbations. In this paper, we study the anti-perturbation ability of the network from the feature maps of convolutional layers. Our theoretical analysis discovers that larger convolutional features before average pooling can contribute to better resistance to perturbations, but the conclusion is not true for max pooling. Based on the theoretical findings, we present two feasible ways to improve the robustness of existing neural networks. The proposed approaches are very simple and only require upsampling the inputs or modifying the stride configuration of convolution operators. We test our approaches on several benchmark neural network architectures, including AlexNet, VGG16, RestNet18 and PreActResNet18, and achieve non-trivial improvements on both natural accuracy and robustness under various attacks. Our study brings new insights into the design of robust neural networks. The code is available at \url{https://github.com/MTandHJ/rcm}.



## **45. AEVA: Black-box Backdoor Detection Using Adversarial Extreme Value Analysis**

cs.LG

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2110.14880v4)

**Authors**: Junfeng Guo, Ang Li, Cong Liu

**Abstracts**: Deep neural networks (DNNs) are proved to be vulnerable against backdoor attacks. A backdoor is often embedded in the target DNNs through injecting a backdoor trigger into training examples, which can cause the target DNNs misclassify an input attached with the backdoor trigger. Existing backdoor detection methods often require the access to the original poisoned training data, the parameters of the target DNNs, or the predictive confidence for each given input, which are impractical in many real-world applications, e.g., on-device deployed DNNs. We address the black-box hard-label backdoor detection problem where the DNN is fully black-box and only its final output label is accessible. We approach this problem from the optimization perspective and show that the objective of backdoor detection is bounded by an adversarial objective. Further theoretical and empirical studies reveal that this adversarial objective leads to a solution with highly skewed distribution; a singularity is often observed in the adversarial map of a backdoor-infected example, which we call the adversarial singularity phenomenon. Based on this observation, we propose the adversarial extreme value analysis(AEVA) to detect backdoors in black-box neural networks. AEVA is based on an extreme value analysis of the adversarial map, computed from the monte-carlo gradient estimation. Evidenced by extensive experiments across multiple popular tasks and backdoor attacks, our approach is shown effective in detecting backdoor attacks under the black-box hard-label scenarios.



## **46. Bounding Membership Inference**

cs.LG

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2202.12232v1)

**Authors**: Anvith Thudi, Ilia Shumailov, Franziska Boenisch, Nicolas Papernot

**Abstracts**: Differential Privacy (DP) is the de facto standard for reasoning about the privacy guarantees of a training algorithm. Despite the empirical observation that DP reduces the vulnerability of models to existing membership inference (MI) attacks, a theoretical underpinning as to why this is the case is largely missing in the literature. In practice, this means that models need to be trained with DP guarantees that greatly decrease their accuracy. In this paper, we provide a tighter bound on the accuracy of any MI adversary when a training algorithm provides $\epsilon$-DP. Our bound informs the design of a novel privacy amplification scheme, where an effective training set is sub-sampled from a larger set prior to the beginning of training, to greatly reduce the bound on MI accuracy. As a result, our scheme enables $\epsilon$-DP users to employ looser DP guarantees when training their model to limit the success of any MI adversary; this ensures that the model's accuracy is less impacted by the privacy guarantee. Finally, we discuss implications of our MI bound on the field of machine unlearning.



## **47. Dynamic Defense Against Byzantine Poisoning Attacks in Federated Learning**

cs.LG

10 pages

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2007.15030v2)

**Authors**: Nuria Rodríguez-Barroso, Eugenio Martínez-Cámara, M. Victoria Luzón, Francisco Herrera

**Abstracts**: Federated learning, as a distributed learning that conducts the training on the local devices without accessing to the training data, is vulnerable to Byzatine poisoning adversarial attacks. We argue that the federated learning model has to avoid those kind of adversarial attacks through filtering out the adversarial clients by means of the federated aggregation operator. We propose a dynamic federated aggregation operator that dynamically discards those adversarial clients and allows to prevent the corruption of the global learning model. We assess it as a defense against adversarial attacks deploying a deep learning classification model in a federated learning setting on the Fed-EMNIST Digits, Fashion MNIST and CIFAR-10 image datasets. The results show that the dynamic selection of the clients to aggregate enhances the performance of the global learning model and discards the adversarial and poor (with low quality models) clients.



## **48. HODA: Hardness-Oriented Detection of Model Extraction Attacks**

cs.LG

15 pages, 12 figures, 7 tables, 2 Alg

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2106.11424v2)

**Authors**: Amir Mahdi Sadeghzadeh, Amir Mohammad Sobhanian, Faezeh Dehghan, Rasool Jalili

**Abstracts**: Model Extraction attacks exploit the target model's prediction API to create a surrogate model in order to steal or reconnoiter the functionality of the target model in the black-box setting. Several recent studies have shown that a data-limited adversary who has no or limited access to the samples from the target model's training data distribution can use synthesis or semantically similar samples to conduct model extraction attacks. In this paper, we define the hardness degree of a sample using the concept of learning difficulty. The hardness degree of a sample depends on the epoch number that the predicted label of that sample converges. We investigate the hardness degree of samples and demonstrate that the hardness degree histogram of a data-limited adversary's sample sequences is distinguishable from the hardness degree histogram of benign users' samples sequences. We propose Hardness-Oriented Detection Approach (HODA) to detect the sample sequences of model extraction attacks. The results demonstrate that HODA can detect the sample sequences of model extraction attacks with a high success rate by only monitoring 100 samples of them, and it outperforms all previous model extraction detection methods.



## **49. Feature Importance-aware Transferable Adversarial Attacks**

cs.CV

Accepted to ICCV 2021

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2107.14185v3)

**Authors**: Zhibo Wang, Hengchang Guo, Zhifei Zhang, Wenxin Liu, Zhan Qin, Kui Ren

**Abstracts**: Transferability of adversarial examples is of central importance for attacking an unknown model, which facilitates adversarial attacks in more practical scenarios, e.g., black-box attacks. Existing transferable attacks tend to craft adversarial examples by indiscriminately distorting features to degrade prediction accuracy in a source model without aware of intrinsic features of objects in the images. We argue that such brute-force degradation would introduce model-specific local optimum into adversarial examples, thus limiting the transferability. By contrast, we propose the Feature Importance-aware Attack (FIA), which disrupts important object-aware features that dominate model decisions consistently. More specifically, we obtain feature importance by introducing the aggregate gradient, which averages the gradients with respect to feature maps of the source model, computed on a batch of random transforms of the original clean image. The gradients will be highly correlated to objects of interest, and such correlation presents invariance across different models. Besides, the random transforms will preserve intrinsic features of objects and suppress model-specific information. Finally, the feature importance guides to search for adversarial examples towards disrupting critical features, achieving stronger transferability. Extensive experimental evaluation demonstrates the effectiveness and superior performance of the proposed FIA, i.e., improving the success rate by 9.5% against normally trained models and 12.8% against defense models as compared to the state-of-the-art transferable attacks. Code is available at: https://github.com/hcguoO0/FIA



## **50. Robust Probabilistic Time Series Forecasting**

cs.LG

AISTATS 2022 camera ready version

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2202.11910v1)

**Authors**: TaeHo Yoon, Youngsuk Park, Ernest K. Ryu, Yuyang Wang

**Abstracts**: Probabilistic time series forecasting has played critical role in decision-making processes due to its capability to quantify uncertainties. Deep forecasting models, however, could be prone to input perturbations, and the notion of such perturbations, together with that of robustness, has not even been completely established in the regime of probabilistic forecasting. In this work, we propose a framework for robust probabilistic time series forecasting. First, we generalize the concept of adversarial input perturbations, based on which we formulate the concept of robustness in terms of bounded Wasserstein deviation. Then we extend the randomized smoothing technique to attain robust probabilistic forecasters with theoretical robustness certificates against certain classes of adversarial perturbations. Lastly, extensive experiments demonstrate that our methods are empirically effective in enhancing the forecast quality under additive adversarial attacks and forecast consistency under supplement of noisy observations.



